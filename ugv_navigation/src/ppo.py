#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


torch.autograd.set_detect_anomaly(True)


class RolloutBuffer:
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.memory = deque(maxlen=self.max_size)

    def add(self, state1, state2, action_vx, action_vz, log_prob_vx, log_prob_vz, reward, done, value):
        self.memory.append((state1, state2, action_vx, action_vz, log_prob_vx, log_prob_vz, reward, done, value))

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)

    def sample_tensors(self, device):
        states1, states2, action_vx, action_vz, log_prob_vx, log_prob_vz, rewards, dones, values = zip(*self.memory)
        return {
            "states1": torch.FloatTensor(np.stack(states1)).to(device),
            "states2": torch.FloatTensor(np.stack(states2)).to(device),
            "action_vx": torch.LongTensor(np.asarray(action_vx)).to(device),
            "action_vz": torch.LongTensor(np.asarray(action_vz)).to(device),
            "old_log_probs": torch.FloatTensor(np.asarray(log_prob_vx) + np.asarray(log_prob_vz)).to(device),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
            "values": np.asarray(values, dtype=np.float32),
        }


class PPONet(nn.Module):
    def __init__(self, action_space_vx, action_space_vz):
        super(PPONet, self).__init__()
        self.action_space_vx = action_space_vx
        self.action_space_vz = action_space_vz

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
        self.value_head = nn.Linear(256, 1)

    def forward(self, state1, state2):
        batch_size = state1.size(0)
        img = state1 / 255
        x3 = self.cnn_a(img.transpose(1, 3))

        x_target = F.relu(self.fc_target(state2))
        x_merge = torch.cat((x3.view(batch_size, -1), x_target), axis=1)
        fc_1 = F.relu(self.fc_1(x_merge))
        fc_2 = F.relu(self.fc_2(fc_1))

        vx_logits = self.output_vx(fc_2)
        vz_logits = self.output_vz(fc_2)
        value = self.value_head(fc_2).squeeze(-1)
        return vx_logits, vz_logits, value


class PPOPolicyExport(nn.Module):
    def __init__(self, policy):
        super(PPOPolicyExport, self).__init__()
        self.policy = policy

    def forward(self, state1, state2):
        vx_logits, vz_logits, _ = self.policy(state1, state2)
        return vx_logits, vz_logits


class PPO:
    def __init__(
        self,
        env,
        action_space_vx,
        action_space_vz,
        learning_rate=1e-4,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        update_epochs=4,
        max_grad_norm=0.5,
        memory_size=100000,
    ):
        super(PPO, self).__init__()
        self.env = env
        self.action_space_vx = action_space_vx
        self.action_space_vz = action_space_vz

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(
            "Using Device:",
            torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU",
        )

        self.policy = PPONet(action_space_vx=self.action_space_vx, action_space_vz=self.action_space_vz).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.rollout_buffer = RolloutBuffer(memory_size)

        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm
        print("PPO Baseline Initialized")

    def get_action(self, state1, state2, deterministic=False):
        self.policy.eval()
        with torch.no_grad():
            state1_tensor = torch.FloatTensor(state1).to(self.device).unsqueeze(0)
            state2_tensor = torch.FloatTensor(state2).to(self.device).unsqueeze(0)
            vx_logits, vz_logits, value = self.policy(state1_tensor, state2_tensor)

            if deterministic:
                action_vx = torch.argmax(vx_logits, dim=1)
                action_vz = torch.argmax(vz_logits, dim=1)
                log_prob_vx = F.log_softmax(vx_logits, dim=1).gather(1, action_vx.view(-1, 1)).squeeze(1)
                log_prob_vz = F.log_softmax(vz_logits, dim=1).gather(1, action_vz.view(-1, 1)).squeeze(1)
            else:
                dist_vx = Categorical(logits=vx_logits)
                dist_vz = Categorical(logits=vz_logits)
                action_vx = dist_vx.sample()
                action_vz = dist_vz.sample()
                log_prob_vx = dist_vx.log_prob(action_vx)
                log_prob_vz = dist_vz.log_prob(action_vz)

        return (
            int(action_vx.item()),
            int(action_vz.item()),
            float(log_prob_vx.item()),
            float(log_prob_vz.item()),
            float(value.item()),
        )

    def get_value(self, state1, state2):
        self.policy.eval()
        with torch.no_grad():
            state1_tensor = torch.FloatTensor(state1).to(self.device).unsqueeze(0)
            state2_tensor = torch.FloatTensor(state2).to(self.device).unsqueeze(0)
            _, _, value = self.policy(state1_tensor, state2_tensor)
        return float(value.item())

    def evaluate_actions(self, states1, states2, action_vx, action_vz):
        vx_logits, vz_logits, values = self.policy(states1, states2)
        dist_vx = Categorical(logits=vx_logits)
        dist_vz = Categorical(logits=vz_logits)

        log_probs = dist_vx.log_prob(action_vx) + dist_vz.log_prob(action_vz)
        entropy = dist_vx.entropy() + dist_vz.entropy()
        return log_probs, entropy, values

    def compute_returns_and_advantages(self, rewards, dones, values, next_value):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_values = values[step + 1]

            delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[step] = gae

        returns = advantages + values
        return returns, advantages

    def learn(self, next_value=0.0):
        if len(self.rollout_buffer) == 0:
            return None

        self.policy.train()
        tensors = self.rollout_buffer.sample_tensors(self.device)
        returns, advantages = self.compute_returns_and_advantages(
            tensors["rewards"], tensors["dones"], tensors["values"], next_value
        )

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        num_samples = tensors["states1"].size(0)
        batch_size = min(self.batch_size, num_samples)
        loss_summaries = []

        for _ in range(self.update_epochs):
            indices = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, batch_size):
                batch_indices = indices[start : start + batch_size]

                log_probs, entropy, values = self.evaluate_actions(
                    tensors["states1"][batch_indices],
                    tensors["states2"][batch_indices],
                    tensors["action_vx"][batch_indices],
                    tensors["action_vz"][batch_indices],
                )

                ratio = torch.exp(log_probs - tensors["old_log_probs"][batch_indices])
                surr1 = ratio * advantages[batch_indices]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[batch_indices]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns[batch_indices])
                entropy_loss = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                loss_summaries.append(
                    (
                        float(policy_loss.item()),
                        float(value_loss.item()),
                        float(entropy_loss.item()),
                        float(loss.item()),
                    )
                )

        self.rollout_buffer.clear()
        mean_losses = np.asarray(loss_summaries, dtype=np.float32).mean(axis=0)
        return {
            "policy_loss": float(mean_losses[0]),
            "value_loss": float(mean_losses[1]),
            "entropy": float(mean_losses[2]),
            "total_loss": float(mean_losses[3]),
        }

    def save_model(self, path):
        checkpoint = {
            "model_states": self.policy.state_dict(),
            "optimizer_states": self.optimizer.state_dict(),
            "action_space_vx": self.action_space_vx,
            "action_space_vz": self.action_space_vz,
        }
        torch.save(checkpoint, path)

    def save_onnx_model(self, param_path_onnx):
        self.policy.eval()
        export_policy = PPOPolicyExport(self.policy).to(self.device)
        dummy_state1 = torch.randn(64, 224, 224, 12).to(self.device)
        dummy_state2 = torch.randn(64, 2).to(self.device)
        torch.onnx.export(
            export_policy,
            (dummy_state1, dummy_state2),
            param_path_onnx,
            input_names=["dummy_state1", "dummy_state2"],
            output_names=["output_velocity_x", "output_velocity_y"],
            dynamic_axes={"dummy_state1": {0: "batch_size"}, "dummy_state2": {0: "batch_size"}},
        )

    def load_model(self, filename, device):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.policy.load_state_dict(checkpoint["model_states"])
            self.optimizer.load_state_dict(checkpoint["optimizer_states"])
            self.policy.eval()
            print(f"Model and optimizer states have been loaded from {filename}")
        else:
            print(f"No file found at {filename}, unable to load states.")
