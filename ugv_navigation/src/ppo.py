#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import os
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


if os.environ.get("PPO_DETECT_ANOMALY") == "1":
    torch.autograd.set_detect_anomaly(True)


class RolloutBuffer:
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.memory = deque(maxlen=self.max_size)
        self.path_ends = []
        self._last_finished_index = 0

    def add(self, state1, state2, action_vx, action_vz, log_prob_vx, log_prob_vz, reward, done, value):
        self.memory.append((state1, state2, action_vx, action_vz, log_prob_vx, log_prob_vz, reward, done, value))

    def finish_path(self, next_value=0.0):
        end_index = len(self.memory)
        if end_index == self._last_finished_index:
            return
        self.path_ends.append((end_index, float(next_value)))
        self._last_finished_index = end_index

    def clear(self):
        self.memory.clear()
        self.path_ends.clear()
        self._last_finished_index = 0

    def __len__(self):
        return len(self.memory)

    def sample_tensors(self):
        states1, states2, action_vx, action_vz, log_prob_vx, log_prob_vz, rewards, dones, values = zip(*self.memory)
        return {
            "states1": np.stack(states1),
            "states2": np.asarray(states2, dtype=np.float32),
            "action_vx": np.asarray(action_vx, dtype=np.int64),
            "action_vz": np.asarray(action_vz, dtype=np.int64),
            "old_log_probs": np.asarray(log_prob_vx, dtype=np.float32) + np.asarray(log_prob_vz, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
            "values": np.asarray(values, dtype=np.float32),
            "path_ends": list(self.path_ends),
        }


class PPONet(nn.Module):
    def __init__(self, action_space_vx, action_space_vz, normalize_state2=True):
        super(PPONet, self).__init__()
        self.action_space_vx = action_space_vx
        self.action_space_vz = action_space_vz
        self.normalize_state2 = normalize_state2

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
        self._init_weights()

    def _init_weights(self):
        relu_gain = nn.init.calculate_gain("relu")
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=relu_gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        for policy_head in (self.output_vx, self.output_vz):
            nn.init.orthogonal_(policy_head.weight, gain=0.01)
            nn.init.constant_(policy_head.bias, 0.0)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, state1, state2):
        batch_size = state1.size(0)
        img = state1 / 255
        x3 = self.cnn_a(img.transpose(1, 3))

        if self.normalize_state2:
            state2 = state2 / state2.new_tensor([30.0, math.pi])
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
        target_kl=0.03,
        use_value_clip=True,
        normalize_state2=True,
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

        self.policy = PPONet(
            action_space_vx=self.action_space_vx,
            action_space_vz=self.action_space_vz,
            normalize_state2=normalize_state2,
        ).to(self.device)
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
        self.target_kl = target_kl
        self.use_value_clip = use_value_clip
        self.normalize_state2 = normalize_state2
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

    def compute_returns_and_advantages(self, rewards, dones, values, next_value, path_ends=None):
        advantages = np.zeros_like(rewards, dtype=np.float32)

        if path_ends is None:
            path_ends = []
        path_ends = list(path_ends)
        if not path_ends or path_ends[-1][0] < len(rewards):
            path_ends.append((len(rewards), float(next_value)))

        path_start = 0
        for path_end, bootstrap_value in path_ends:
            gae = 0.0
            for step in reversed(range(path_start, path_end)):
                if step == path_end - 1:
                    next_non_terminal = 1.0 - dones[step]
                    next_values = bootstrap_value
                else:
                    next_non_terminal = 1.0 - dones[step]
                    next_values = values[step + 1]

                delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                advantages[step] = gae
            path_start = path_end

        returns = advantages + values
        return returns, advantages

    def _batch_tensor(self, array, batch_indices, dtype):
        return torch.as_tensor(array[batch_indices], dtype=dtype, device=self.device)

    @staticmethod
    def _explained_variance(pred_values, returns):
        returns_variance = torch.var(returns, unbiased=False)
        if returns_variance.item() < 1e-8:
            return 0.0
        return float((1.0 - torch.var(returns - pred_values, unbiased=False) / returns_variance).item())

    def learn(self, next_value=0.0):
        if len(self.rollout_buffer) == 0:
            return None

        self.policy.train()
        tensors = self.rollout_buffer.sample_tensors()
        returns, advantages = self.compute_returns_and_advantages(
            tensors["rewards"],
            tensors["dones"],
            tensors["values"],
            next_value,
            tensors["path_ends"],
        )

        returns = torch.as_tensor(returns, dtype=torch.float32)
        advantages = torch.as_tensor(advantages, dtype=torch.float32)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        old_values = torch.as_tensor(tensors["values"], dtype=torch.float32)
        num_samples = tensors["states1"].shape[0]
        batch_size = min(self.batch_size, num_samples)
        loss_summaries = []
        stop_training = False

        for _ in range(self.update_epochs):
            indices = np.random.permutation(num_samples)
            for start in range(0, num_samples, batch_size):
                batch_indices = indices[start : start + batch_size]
                batch_tensor_indices = torch.as_tensor(batch_indices, dtype=torch.long)
                states1_batch = self._batch_tensor(tensors["states1"], batch_indices, torch.float32)
                states2_batch = self._batch_tensor(tensors["states2"], batch_indices, torch.float32)
                action_vx_batch = self._batch_tensor(tensors["action_vx"], batch_indices, torch.long)
                action_vz_batch = self._batch_tensor(tensors["action_vz"], batch_indices, torch.long)
                old_log_probs_batch = self._batch_tensor(tensors["old_log_probs"], batch_indices, torch.float32)
                returns_batch = returns[batch_tensor_indices].to(self.device)
                advantages_batch = advantages[batch_tensor_indices].to(self.device)
                old_values_batch = old_values[batch_tensor_indices].to(self.device)

                log_probs, entropy, values = self.evaluate_actions(
                    states1_batch,
                    states2_batch,
                    action_vx_batch,
                    action_vz_batch,
                )

                log_ratio = log_probs - old_log_probs_batch
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                if self.use_value_clip:
                    values_clipped = old_values_batch + torch.clamp(
                        values - old_values_batch,
                        -self.clip_eps,
                        self.clip_eps,
                    )
                    value_loss_unclipped = (values - returns_batch).pow(2)
                    value_loss_clipped = (values_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = F.mse_loss(values, returns_batch)
                entropy_loss = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > self.clip_eps).float().mean()

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
                        float(approx_kl.detach().item()),
                        float(clip_fraction.detach().item()),
                    )
                )

                if self.target_kl is not None and float(approx_kl.detach().item()) > self.target_kl:
                    stop_training = True
                    break
            if stop_training:
                break

        self.policy.eval()
        with torch.no_grad():
            predicted_values = []
            for start in range(0, num_samples, batch_size):
                batch_indices = np.arange(start, min(start + batch_size, num_samples))
                states1_batch = self._batch_tensor(tensors["states1"], batch_indices, torch.float32)
                states2_batch = self._batch_tensor(tensors["states2"], batch_indices, torch.float32)
                _, _, values = self.policy(states1_batch, states2_batch)
                predicted_values.append(values.cpu())
            predicted_values = torch.cat(predicted_values)
        explained_variance = self._explained_variance(predicted_values, returns)

        self.rollout_buffer.clear()
        mean_losses = np.asarray(loss_summaries, dtype=np.float32).mean(axis=0)
        return {
            "policy_loss": float(mean_losses[0]),
            "value_loss": float(mean_losses[1]),
            "entropy": float(mean_losses[2]),
            "total_loss": float(mean_losses[3]),
            "approx_kl": float(mean_losses[4]),
            "clip_fraction": float(mean_losses[5]),
            "explained_variance": float(explained_variance),
        }

    def save_model(self, path):
        checkpoint = {
            "model_states": self.policy.state_dict(),
            "optimizer_states": self.optimizer.state_dict(),
            "action_space_vx": self.action_space_vx,
            "action_space_vz": self.action_space_vz,
            "normalize_state2": self.normalize_state2,
            "use_value_clip": self.use_value_clip,
            "target_kl": self.target_kl,
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
