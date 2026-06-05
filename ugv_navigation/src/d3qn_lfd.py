#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import os
import random
from collections import deque

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx


class DQfDReplayBuffer:
    def __init__(
        self,
        demo_capacity=20000,
        agent_capacity=10000,
        per_alpha=0.6,
        beta_start=0.4,
        beta_frames=200000,
        demo_priority_bonus=1.0,
        priority_eps=1e-6,
    ):
        self.demo_capacity = demo_capacity
        self.agent_capacity = agent_capacity
        self.per_alpha = per_alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.demo_priority_bonus = demo_priority_bonus
        self.priority_eps = priority_eps

        self.demo_memory = []
        self.demo_priorities = []
        self.agent_memory = []
        self.agent_priorities = []
        self.agent_next_idx = 0
        self.beta_step = 0

    def __len__(self):
        return len(self.demo_memory) + len(self.agent_memory)

    def max_priority(self):
        priorities = self.demo_priorities + self.agent_priorities
        if not priorities:
            return 1.0
        return max(priorities)

    def add(self, entry):
        priority = self.max_priority()
        if entry["is_demo"]:
            if len(self.demo_memory) >= self.demo_capacity:
                return
            self.demo_memory.append(entry)
            self.demo_priorities.append(priority)
            return

        if len(self.agent_memory) < self.agent_capacity:
            self.agent_memory.append(entry)
            self.agent_priorities.append(priority)
            return

        self.agent_memory[self.agent_next_idx] = entry
        self.agent_priorities[self.agent_next_idx] = priority
        self.agent_next_idx = (self.agent_next_idx + 1) % self.agent_capacity

    def beta(self):
        progress = min(1.0, self.beta_step / float(max(1, self.beta_frames)))
        return self.beta_start + progress * (1.0 - self.beta_start)

    def _combined_entries(self):
        refs = []
        entries = []
        priorities = []

        for index, entry in enumerate(self.demo_memory):
            refs.append(("demo", index))
            entries.append(entry)
            priorities.append(self.demo_priorities[index] + self.demo_priority_bonus)

        for index, entry in enumerate(self.agent_memory):
            refs.append(("agent", index))
            entries.append(entry)
            priorities.append(self.agent_priorities[index])

        return refs, entries, np.asarray(priorities, dtype=np.float64)

    def sample(self, batch_size):
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        self.beta_step += 1
        refs, entries, priorities = self._combined_entries()
        scaled = np.power(np.maximum(priorities, self.priority_eps), self.per_alpha)
        probabilities = scaled / scaled.sum()
        sampled_indices = np.random.choice(len(entries), size=batch_size, replace=True, p=probabilities)

        sampled_entries = [entries[index] for index in sampled_indices]
        sampled_refs = [refs[index] for index in sampled_indices]
        sampled_probs = probabilities[sampled_indices]
        weights = np.power(len(entries) * sampled_probs, -self.beta())
        weights = weights / weights.max()
        return sampled_entries, sampled_refs, weights.astype(np.float32), self.beta()

    def update_priorities(self, refs, priorities):
        for ref, priority in zip(refs, priorities):
            priority = float(abs(priority)) + self.priority_eps
            kind, index = ref
            if kind == "demo":
                self.demo_priorities[index] = priority
            else:
                self.agent_priorities[index] = priority


class NStepTransitionBuilder:
    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.pending = deque()

    def append(self, transition):
        self.pending.append(transition)
        entries = []

        if len(self.pending) >= self.n_step:
            entries.append(self._build_entry(self.n_step))
            self.pending.popleft()

        if transition["episode_end"]:
            entries.extend(self.flush())

        return entries

    def flush(self):
        entries = []
        while self.pending:
            entries.append(self._build_entry(len(self.pending)))
            self.pending.popleft()
        return entries

    def _build_entry(self, horizon):
        first = self.pending[0]
        n_reward = 0.0
        n_done = False
        n_steps = 0
        n_next_state1 = first["next_state1"]
        n_next_state2 = first["next_state2"]

        for index in range(horizon):
            transition = self.pending[index]
            n_reward += (self.gamma**index) * transition["reward"]
            n_steps += 1
            n_next_state1 = transition["next_state1"]
            n_next_state2 = transition["next_state2"]
            n_done = transition["done"]
            if transition["done"] or transition["episode_end"]:
                break

        entry = dict(first)
        entry["n_reward"] = n_reward
        entry["n_next_state1"] = n_next_state1
        entry["n_next_state2"] = n_next_state2
        entry["n_done"] = n_done
        entry["n_steps"] = n_steps
        entry.pop("episode_end", None)
        return entry


class DQNNet(nn.Module):
    def __init__(self, network, action_space_vx, action_space_vz):
        super(DQNNet, self).__init__()
        self.action_space_vx = action_space_vx
        self.action_space_vz = action_space_vz
        self.network = network

        self.fc_target = nn.Linear(2, 64)
        self.cnn_a = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=(4, 3), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 3), stride=3),
            nn.ReLU(),
        )

        self.fc_1 = nn.Linear(64 * 18 * 18 + 64, 256)
        self.fc_2 = nn.Linear(256, 256)
        self.output_vx = nn.Linear(256, len(self.action_space_vx))
        self.output_vz = nn.Linear(256, len(self.action_space_vz))

        self.advantage_vx = nn.Linear(128, len(self.action_space_vx))
        self.value_vx = nn.Linear(128, 1)
        self.advantage_vz = nn.Linear(128, len(self.action_space_vz))
        self.value_vz = nn.Linear(128, 1)

    def forward(self, state1, state2):
        batch_size = state1.size(0)
        img = state1 / 255
        x3 = self.cnn_a(img.transpose(1, 3))

        x_target = F.relu(self.fc_target(state2))
        x_merge = torch.cat((x3.view(batch_size, -1), x_target), dim=1)
        fc_1 = F.relu(self.fc_1(x_merge))
        fc_2 = F.relu(self.fc_2(fc_1))

        if self.network == "Duel":
            advantage_vx, value_vx = torch.split(fc_2, 128, dim=1)
            advantage_vx = self.advantage_vx(advantage_vx)
            value_vx = self.value_vx(value_vx)
            vx_output = value_vx + advantage_vx - torch.mean(advantage_vx, dim=1, keepdim=True)

            advantage_vz, value_vz = torch.split(fc_2, 128, dim=1)
            advantage_vz = self.advantage_vz(advantage_vz)
            value_vz = self.value_vz(value_vz)
            vz_output = value_vz + advantage_vz - torch.mean(advantage_vz, dim=1, keepdim=True)
        else:
            vx_output = self.output_vx(fc_2)
            vz_output = self.output_vz(fc_2)

        return vx_output, vz_output


class D3QNLfD:
    def __init__(
        self,
        env,
        action_space_vx,
        action_space_vz,
        demo_capacity=20000,
        agent_capacity=10000,
        learning_rate=1e-4,
        batch_size=64,
        target_update=1000,
        gamma=0.99,
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
    ):
        super(D3QNLfD, self).__init__()
        self.env = env
        self.network = network
        self.action_space_vx = action_space_vx
        self.action_space_vz = action_space_vz
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(
            "Using Device:",
            torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU",
        )

        self.predict_net = DQNNet(network, action_space_vx, action_space_vz).to(self.device)
        self.target_net = DQNNet(network, action_space_vx, action_space_vz).to(self.device)
        self.target_net.load_state_dict(self.predict_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.predict_net.parameters(), lr=learning_rate)

        self.replay_buffer = DQfDReplayBuffer(
            demo_capacity=demo_capacity,
            agent_capacity=agent_capacity,
            per_alpha=per_alpha,
            beta_start=beta_start,
            beta_frames=beta_frames,
            demo_priority_bonus=demo_priority_bonus,
        )
        self.demo_builder = NStepTransitionBuilder(n_step, gamma)
        self.agent_builder = NStepTransitionBuilder(n_step, gamma)

        self.batch_size = batch_size
        self.target_update = target_update
        self.update_count = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_period = epsilon_period
        self.n_step = n_step
        self.margin = margin
        self.lambda_n = lambda_n
        self.lambda_margin = lambda_margin
        self.lambda_l2 = lambda_l2
        print("D3QN-LfD Initialized")

    def get_action(self, state1, state2):
        if np.random.random() < self.epsilon:
            return (
                np.random.randint(0, len(self.action_space_vx)),
                np.random.randint(0, len(self.action_space_vz)),
            )

        self.predict_net.eval()
        with torch.no_grad():
            state1_tensor = torch.FloatTensor(state1).to(self.device).unsqueeze(0)
            state2_tensor = torch.FloatTensor(state2).to(self.device).unsqueeze(0)
            q_values_vx, q_values_vz = self.predict_net(state1_tensor, state2_tensor)
            action_vx_index = np.argmax(q_values_vx.cpu().detach().numpy())
            action_vz_index = np.argmax(q_values_vz.cpu().detach().numpy())
        return action_vx_index, action_vz_index

    def store_transition(
        self,
        state1,
        state2,
        action_index,
        expert_index,
        reward,
        next_state1,
        next_state2,
        done,
        episode_end,
        is_demo,
    ):
        transition = {
            "state1": state1,
            "state2": state2,
            "action_index": action_index,
            "expert_index": expert_index,
            "reward": float(reward),
            "next_state1": next_state1,
            "next_state2": next_state2,
            "done": bool(done),
            "episode_end": bool(episode_end),
            "is_demo": bool(is_demo),
        }
        builder = self.demo_builder if is_demo else self.agent_builder
        for entry in builder.append(transition):
            self.replay_buffer.add(entry)

    def _batch_to_tensors(self, entries, weights):
        states1 = torch.FloatTensor(np.stack([entry["state1"] for entry in entries])).to(self.device)
        states2 = torch.FloatTensor(np.stack([entry["state2"] for entry in entries])).to(self.device)
        next_states1 = torch.FloatTensor(np.stack([entry["next_state1"] for entry in entries])).to(self.device)
        next_states2 = torch.FloatTensor(np.stack([entry["next_state2"] for entry in entries])).to(self.device)
        n_next_states1 = torch.FloatTensor(np.stack([entry["n_next_state1"] for entry in entries])).to(self.device)
        n_next_states2 = torch.FloatTensor(np.stack([entry["n_next_state2"] for entry in entries])).to(self.device)

        action_indices = torch.LongTensor(np.stack([entry["action_index"] for entry in entries])).to(self.device)
        expert_indices = torch.LongTensor(np.stack([entry["expert_index"] for entry in entries])).to(self.device)
        rewards = torch.FloatTensor(np.asarray([entry["reward"] for entry in entries], dtype=np.float32)).to(self.device)
        dones = torch.FloatTensor(np.asarray([entry["done"] for entry in entries], dtype=np.float32)).to(self.device)
        n_rewards = torch.FloatTensor(np.asarray([entry["n_reward"] for entry in entries], dtype=np.float32)).to(
            self.device
        )
        n_dones = torch.FloatTensor(np.asarray([entry["n_done"] for entry in entries], dtype=np.float32)).to(self.device)
        n_steps = torch.FloatTensor(np.asarray([entry["n_steps"] for entry in entries], dtype=np.float32)).to(
            self.device
        )
        is_demo = torch.BoolTensor([entry["is_demo"] for entry in entries]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device).view(-1, 1)

        return {
            "states1": states1,
            "states2": states2,
            "next_states1": next_states1,
            "next_states2": next_states2,
            "n_next_states1": n_next_states1,
            "n_next_states2": n_next_states2,
            "action_vx": action_indices[:, 0].long().view(-1, 1),
            "action_vz": action_indices[:, 1].long().view(-1, 1),
            "expert_vx": expert_indices[:, 0].long().view(-1, 1),
            "expert_vz": expert_indices[:, 1].long().view(-1, 1),
            "rewards": rewards.view(-1, 1),
            "dones": dones.view(-1, 1),
            "n_rewards": n_rewards.view(-1, 1),
            "n_dones": n_dones.view(-1, 1),
            "n_steps": n_steps.view(-1, 1),
            "is_demo": is_demo,
            "weights": weights,
        }

    def _double_dqn_targets(self, next_states1, next_states2, rewards, dones, discount):
        with torch.no_grad():
            next_q_vx, next_q_vz = self.predict_net(next_states1, next_states2)
            _, next_action_vx = torch.max(next_q_vx, 1)
            _, next_action_vz = torch.max(next_q_vz, 1)
            target_q_vx, target_q_vz = self.target_net(next_states1, next_states2)
            target_vx = target_q_vx.gather(1, next_action_vx.view(-1, 1))
            target_vz = target_q_vz.gather(1, next_action_vz.view(-1, 1))
            target_vx = rewards + discount * target_vx * (1.0 - dones)
            target_vz = rewards + discount * target_vz * (1.0 - dones)
        return target_vx, target_vz

    def _large_margin_loss(self, q_values, expert_actions, is_demo):
        if not torch.any(is_demo):
            return q_values.new_tensor(0.0), q_values.new_zeros(q_values.size(0), 1)

        margins = torch.full_like(q_values, self.margin)
        margins.scatter_(1, expert_actions, 0.0)
        expert_q = q_values.gather(1, expert_actions)
        per_sample = torch.max(q_values + margins, dim=1, keepdim=True)[0] - expert_q
        per_sample = per_sample * is_demo.float().view(-1, 1)
        masked = per_sample[is_demo.view(-1)]
        return masked.mean(), per_sample.detach()

    def learn(self):
        if len(self.replay_buffer) == 0:
            return None

        entries, refs, weights, beta = self.replay_buffer.sample(self.batch_size)
        batch = self._batch_to_tensors(entries, weights)
        self.predict_net.train()

        current_q_vx, current_q_vz = self.predict_net(batch["states1"], batch["states2"])
        pred_vx = current_q_vx.gather(1, batch["action_vx"])
        pred_vz = current_q_vz.gather(1, batch["action_vz"])

        target_vx, target_vz = self._double_dqn_targets(
            batch["next_states1"],
            batch["next_states2"],
            batch["rewards"],
            batch["dones"],
            self.gamma,
        )
        n_discount = torch.pow(torch.full_like(batch["n_steps"], self.gamma), batch["n_steps"])
        n_target_vx, n_target_vz = self._double_dqn_targets(
            batch["n_next_states1"],
            batch["n_next_states2"],
            batch["n_rewards"],
            batch["n_dones"],
            n_discount,
        )

        td_error_vx = pred_vx - target_vx
        td_error_vz = pred_vz - target_vz
        n_td_error_vx = pred_vx - n_target_vx
        n_td_error_vz = pred_vz - n_target_vz
        td_loss = (batch["weights"] * (td_error_vx.pow(2) + td_error_vz.pow(2))).mean()
        n_step_loss = (batch["weights"] * (n_td_error_vx.pow(2) + n_td_error_vz.pow(2))).mean()

        margin_loss_vx, margin_per_vx = self._large_margin_loss(current_q_vx, batch["expert_vx"], batch["is_demo"])
        margin_loss_vz, margin_per_vz = self._large_margin_loss(current_q_vz, batch["expert_vz"], batch["is_demo"])
        margin_loss = margin_loss_vx + margin_loss_vz

        l2_loss = torch.zeros((), device=self.device)
        for param in self.predict_net.parameters():
            l2_loss = l2_loss + param.pow(2).sum()
        loss = td_loss + self.lambda_n * n_step_loss + self.lambda_margin * margin_loss + self.lambda_l2 * l2_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        priority_errors = (
            td_error_vx.detach().abs()
            + td_error_vz.detach().abs()
            + self.lambda_n * (n_td_error_vx.detach().abs() + n_td_error_vz.detach().abs())
            + self.lambda_margin * (margin_per_vx + margin_per_vz)
        )
        self.replay_buffer.update_priorities(refs, priority_errors.view(-1).cpu().numpy())

        self.update_count += 1
        if self.update_count == self.target_update:
            self.target_net.load_state_dict(self.predict_net.state_dict())
            self.update_count = 0

        demo_fraction = float(batch["is_demo"].float().mean().item())
        return {
            "loss_total": float(loss.item()),
            "loss_td": float(td_loss.item()),
            "loss_n_step": float(n_step_loss.item()),
            "loss_margin": float(margin_loss.item()),
            "loss_l2": float(l2_loss.item()),
            "demo_fraction": demo_fraction,
            "beta": float(beta),
        }

    def save_model(self, path):
        checkpoint = {
            "model_states": self.predict_net.state_dict(),
            "target_model_states": self.target_net.state_dict(),
            "optimizer_states": self.optimizer.state_dict(),
            "action_space_vx": self.action_space_vx,
            "action_space_vz": self.action_space_vz,
        }
        torch.save(checkpoint, path)

    def save_onnx_model(self, param_path_onnx):
        self.predict_net.eval()
        dummy_state1 = torch.randn(64, 224, 224, 12).to(self.device)
        dummy_state2 = torch.randn(64, 2).to(self.device)
        torch.onnx.export(
            self.predict_net,
            (dummy_state1, dummy_state2),
            param_path_onnx,
            input_names=["dummy_state1", "dummy_state2"],
            output_names=["output_velocity_x", "output_velocity_y"],
            dynamic_axes={"dummy_state1": {0: "batch_size"}, "dummy_state2": {0: "batch_size"}},
        )

    def load_model(self, filename, device):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.predict_net.load_state_dict(checkpoint["model_states"])
            target_states = checkpoint.get("target_model_states", checkpoint["model_states"])
            self.target_net.load_state_dict(target_states)
            self.optimizer.load_state_dict(checkpoint["optimizer_states"])
            self.predict_net.eval()
            self.target_net.eval()
            print(f"Model and optimizer states have been loaded from {filename}")
        else:
            print(f"No file found at {filename}, unable to load states.")

    @staticmethod
    def load_onnx_model(onnx_file_path):
        return ort.InferenceSession(onnx_file_path)
