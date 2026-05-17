#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import os
import random
import time

import env
import numpy as np
import ppo
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
wandb.init(project="OFF-ROAD-UGV", name="PPO_" + time.strftime("%Y-%m-%d %H:%M:%S"))

total_episode = 5000
max_step_per_episode = 100

model_path = "Model/PPO/"
pth_path = model_path + "model.pth"
onnx_path = model_path + "model_ppo.onnx"

os.makedirs(model_path, exist_ok=True)

gazebo_ugv = env.GazeboUGV(max_step=max_step_per_episode)
agent = ppo.PPO(
    gazebo_ugv,
    gazebo_ugv.action_space_vx,
    gazebo_ugv.action_space_vz,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    update_epochs=4,
    max_grad_norm=0.5,
    learning_rate=1e-4,
)

ep_reward_list = []
ep_step_list = []
ep_success_list = []

print("Action Space_vx = ", gazebo_ugv.action_space_vx)
print("Action Space_vz = ", gazebo_ugv.action_space_vz)

for i_episode in range(total_episode + 1):
    state1, state2, dist_normalized = gazebo_ugv.reset()
    current_episode_reward = 0
    terminal = False

    for t in range(max_step_per_episode):
        action_space_vx = gazebo_ugv.action_space_vx
        action_space_vz = gazebo_ugv.action_space_vz

        action_vx_index, action_vz_index, log_prob_vx, log_prob_vz, value = agent.get_action(state1, state2)
        print("action{}:{} {}".format(t + 1, action_space_vx[action_vx_index], action_space_vz[action_vz_index]))

        gazebo_ugv.execute_linear_velocity(action_vx_index, action_vz_index)
        rospy.sleep(0.1)
        next_state1, next_state2, terminal, reward, termination_state = gazebo_ugv.step(time_step=t + 1)
        current_episode_reward += reward

        agent.rollout_buffer.add(
            state1,
            state2,
            action_vx_index,
            action_vz_index,
            log_prob_vx,
            log_prob_vz,
            reward,
            terminal,
            value,
        )

        if terminal:
            if termination_state == "arrival":
                ep_success_list.append(1)
            else:
                ep_success_list.append(0)
            break

        state1 = next_state1
        state2 = next_state2

    next_value = 0.0 if terminal else agent.get_value(state1, state2)
    loss_info = agent.learn(next_value=next_value)

    ep_reward_list.append(current_episode_reward)
    ep_step_list.append(t + 1)

    if len(ep_success_list) >= 50:
        recent_results = ep_success_list[-50:]
        success_rate = sum(recent_results) / 50
    elif ep_success_list:
        success_rate = sum(ep_success_list) / len(ep_success_list)
    else:
        success_rate = 0.0

    print("Episode:{} \t step:{} \t current_episode_reward:{:.2f}".format(i_episode, t + 1, current_episode_reward))

    wandb.log({"Reward": current_episode_reward}, step=i_episode)
    wandb.log({"Step": t + 1}, step=i_episode)
    wandb.log({"Success Rate": success_rate}, step=i_episode)
    if loss_info is not None:
        wandb.log(
            {
                "Loss_Policy": loss_info["policy_loss"],
                "Loss_Value": loss_info["value_loss"],
                "Entropy": loss_info["entropy"],
                "Loss_Total": loss_info["total_loss"],
            },
            step=i_episode,
        )

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
