#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import os
import random
import time

import APF_Vel_ROS
import d3qn_lfd
import env
import numpy as np
import rospy
import torch
import wandb


seed = 4
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

wandb.login()
wandb.init(project="OFF-ROAD-UGV", name="D3QN-LfD_" + time.strftime("%Y-%m-%d %H:%M:%S"))
wandb.define_metric("pretrain_step")
wandb.define_metric("episode")
wandb.define_metric("Pretrain_Loss_Total", step_metric="pretrain_step")
wandb.define_metric("Reward", step_metric="episode")
wandb.define_metric("Step", step_metric="episode")
wandb.define_metric("Success Rate", step_metric="episode")
wandb.define_metric("Online_Loss_Total", step_metric="episode")

total_episode = 2000
max_step_per_episode = 100
demo_episodes = 200
min_demo_episodes = 100
max_demo_attempts = demo_episodes * 5
pretrain_steps = 5000
ugv_mass = 1.48

model_path = "Model/D3QN-LfD/"
pth_path = model_path + "model.pth"
onnx_path = model_path + "model_d3qn_lfd.onnx"
os.makedirs(model_path, exist_ok=True)

gazebo_ugv = env.GazeboUGV(max_step=max_step_per_episode)
agent = d3qn_lfd.D3QNLfD(
    gazebo_ugv,
    gazebo_ugv.action_space_vx,
    gazebo_ugv.action_space_vz,
    batch_size=64,
    demo_capacity=20000,
    agent_capacity=10000,
    target_update=4,
    gamma=0.99,
    learning_rate=1e-4,
    epsilon=1.0,
    epsilon_min=0.1,
    epsilon_period=50,
    network="Duel",
    n_step=10,
    margin=0.8,
    lambda_n=1.0,
    lambda_margin=1.0,
    lambda_l2=1e-5,
    per_alpha=0.6,
    beta_start=0.4,
    beta_frames=200000,
    demo_priority_bonus=1.0,
)

ep_reward_list = []
ep_step_list = []
ep_success_list = []

print("Action Space_vx = ", gazebo_ugv.action_space_vx)
print("Action Space_vz = ", gazebo_ugv.action_space_vz)


def get_apf_action(env_obj):
    target_location = np.array(env_obj.goal)
    current_position = np.array(env_obj.self_state[0:2])
    obs_pos = np.array(env_obj.cylinder_pos)
    att, rep, vx_world, vy_world = APF_Vel_ROS.vel_control(
        target_location=target_location,
        current_position=current_position,
        obs_pos=obs_pos,
        mass=ugv_mass,
        obs_radius=5.0,
    )
    yaw = env_obj.self_state[3]
    linear_cmd, angular_cmd = APF_Vel_ROS.vector_to_ugv_controls(vx_world, vy_world, yaw)
    vx_index = APF_Vel_ROS.fuzzy_map_v_triangular(linear_cmd, env_obj.action_space_vx, strategy="min")
    vz_index = APF_Vel_ROS.fuzzy_map_v_triangular(angular_cmd, env_obj.action_space_vz, strategy="max")
    return vx_index, vz_index


def dqn_done_from_termination(terminal, termination_state):
    return bool(terminal)


print("=" * 60)
print("Collecting D3QN-LfD APF demonstrations")
print("=" * 60)
demo_success_count = 0
demo_attempt_count = 0
while demo_success_count < demo_episodes and demo_attempt_count < max_demo_attempts:
    demo_attempt_count += 1
    state1, state2, dist_normalized = gazebo_ugv.reset()
    episode_demo_transitions = []
    final_termination_state = "running"

    for t in range(max_step_per_episode):
        expert_vx_index, expert_vz_index = get_apf_action(gazebo_ugv)
        gazebo_ugv.execute_linear_velocity(expert_vx_index, expert_vz_index)
        rospy.sleep(0.1)
        next_state1, next_state2, terminal, reward, termination_state = gazebo_ugv.step(time_step=t + 1)
        dqn_done = dqn_done_from_termination(terminal, termination_state)
        expert_index = [expert_vx_index, expert_vz_index]
        episode_demo_transitions.append(
            {
                "state1": state1,
                "state2": state2,
                "action_index": expert_index,
                "expert_index": expert_index,
                "reward": reward,
                "next_state1": next_state1,
                "next_state2": next_state2,
                "done": dqn_done,
                "episode_end": terminal,
                "is_demo": True,
            }
        )

        if terminal:
            final_termination_state = termination_state
            break
        state1 = next_state1
        state2 = next_state2

    if final_termination_state == "arrival":
        for transition in episode_demo_transitions:
            agent.store_transition(**transition)
        demo_success_count += 1
        print(
            "Accepted Demo:{} / {} \t attempts:{} / {} \t replay:{} \t steps:{}".format(
                demo_success_count,
                demo_episodes,
                demo_attempt_count,
                max_demo_attempts,
                len(agent.replay_buffer),
                len(episode_demo_transitions),
            )
        )
    else:
        print(
            "Discarded Demo Attempt:{} / {} \t state:{} \t success:{} / {} \t replay:{}".format(
                demo_attempt_count,
                max_demo_attempts,
                final_termination_state,
                demo_success_count,
                demo_episodes,
                len(agent.replay_buffer),
            )
        )

if demo_success_count < min_demo_episodes:
    raise RuntimeError(
        "Only collected {} successful APF demonstrations after {} attempts, below the minimum {}. "
        "Please check APF behavior or increase max_demo_attempts.".format(
            demo_success_count, max_demo_attempts, min_demo_episodes
        )
    )

if demo_success_count < demo_episodes:
    print(
        "WARNING: Only collected {} / {} successful APF demonstrations after {} attempts. "
        "Continue training with available successful demonstrations.".format(
            demo_success_count, demo_episodes, max_demo_attempts
        )
    )

print("=" * 60)
print("Pretraining D3QN-LfD on demonstration replay")
print("=" * 60)
for pretrain_step in range(pretrain_steps):
    loss_info = agent.learn()
    if loss_info is None:
        continue
    if (pretrain_step + 1) % 100 == 0:
        print(
            "Pretrain Step:{} / {} \t loss_total:{:.4f} \t demo_fraction:{:.2f}".format(
                pretrain_step + 1,
                pretrain_steps,
                loss_info["loss_total"],
                loss_info["demo_fraction"],
            )
        )
        wandb.log(
            {
                "pretrain_step": pretrain_step + 1,
                "Pretrain_Loss_Total": loss_info["loss_total"],
            }
        )

print("=" * 60)
print("Online D3QN-LfD training")
print("=" * 60)
for i_episode in range(total_episode + 1):
    state1, state2, dist_normalized = gazebo_ugv.reset()
    current_episode_reward = 0
    loss_info = None

    for t in range(max_step_per_episode):
        action_vx_index, action_vz_index = agent.get_action(state1, state2)
        print(
            "action{}:{} {}".format(
                t + 1,
                gazebo_ugv.action_space_vx[action_vx_index],
                gazebo_ugv.action_space_vz[action_vz_index],
            )
        )

        gazebo_ugv.execute_linear_velocity(action_vx_index, action_vz_index)
        rospy.sleep(0.1)
        next_state1, next_state2, terminal, reward, termination_state = gazebo_ugv.step(time_step=t + 1)
        current_episode_reward += reward
        dqn_done = dqn_done_from_termination(terminal, termination_state)
        action_index = [action_vx_index, action_vz_index]
        agent.store_transition(
            state1,
            state2,
            action_index,
            action_index,
            reward,
            next_state1,
            next_state2,
            dqn_done,
            terminal,
            is_demo=False,
        )

        if len(agent.replay_buffer) >= agent.batch_size:
            loss_info = agent.learn()

        if terminal:
            ep_success_list.append(1 if termination_state == "arrival" else 0)
            break

        state1 = next_state1
        state2 = next_state2

    if (i_episode + 1) % agent.epsilon_period == 0:
        agent.epsilon = max(agent.epsilon_min, agent.epsilon - 0.05)

    ep_reward_list.append(current_episode_reward)
    ep_step_list.append(t + 1)
    if len(ep_success_list) >= 50:
        success_rate = sum(ep_success_list[-50:]) / 50
    elif ep_success_list:
        success_rate = sum(ep_success_list) / len(ep_success_list)
    else:
        success_rate = 0.0

    print(
        "Episode:{} \t step:{} \t current_episode_reward:{:.2f} \t epsilon:{:.2f}".format(
            i_episode,
            t + 1,
            current_episode_reward,
            agent.epsilon,
        )
    )
    wandb.log({
        "episode": i_episode,
        "Reward": current_episode_reward,
        "Step": t + 1,
        "Success Rate": success_rate,
        "Online_Loss_Total": loss_info["loss_total"] if loss_info else None,
    })

    reward_file_path = model_path + "Reward"
    step_file_path = model_path + "Step"
    success_rate_file_path = model_path + "Success_Rate"

    mode = "a" if os.path.exists(success_rate_file_path) else "w"
    with open(success_rate_file_path, mode) as f:
        f.write(f"{success_rate}\n")

    if (i_episode + 1) % 500 == 0:
        mode = "a" if os.path.exists(reward_file_path) else "w"
        with open(reward_file_path, mode) as f:
            for reward_value in ep_reward_list:
                f.write(f"{reward_value}\n")
        ep_reward_list = []

        mode = "a" if os.path.exists(step_file_path) else "w"
        with open(step_file_path, mode) as f:
            for step_value in ep_step_list:
                f.write(f"{step_value}\n")
        ep_step_list = []

        agent.save_model(pth_path)
        agent.save_onnx_model(onnx_path)
        onnx_file_name = "Model_" + time.strftime("%Y-%m-%d") + f"_{i_episode + 1}.onnx"
        agent.save_onnx_model(model_path + onnx_file_name)

agent.save_model(pth_path)
agent.save_onnx_model(onnx_path)
