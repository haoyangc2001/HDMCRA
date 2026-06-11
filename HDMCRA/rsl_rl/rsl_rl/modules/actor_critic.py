# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Beta, Normal
from torch.nn.modules import rnn

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class EC_EFPPO_ActorCritic(nn.Module):
    """
    EC-EFPPO 三网络架构：Policy + Energy Value + Reach Value。

    移植自 Go2HierarchicalMiniCostReachAvoid/model/actorcritic.py 中的
    Policy_Network 和 Value_Network，并按 Go2 训练配置扩展网络宽度/深度。

    三个子网络完全独立（不共享参数）：
    - self.actor: 策略网络，支持 Gaussian 或 Beta 有界动作分布
    - self.energy_critic: 能量价值网络，输出标量 V(s)
    - self.reach_critic: reach 价值网络，输出标量 h(s)
    """

    def __init__(self, num_actor_obs, num_critic_obs, num_actions,
                 hidden_dim=256, num_hidden_layers=2,
                 init_noise_std=1.0, activation='elu',
                 log_std_min=-5.0, log_std_max=2.0,
                 bounded_actor_mean=False,
                 action_distribution='gaussian', **kwargs):
        if kwargs:
            print("EC_EFPPO_ActorCritic.__init__ got unexpected arguments, "
                  "which will be ignored: " + str([key for key in kwargs.keys()]))
        super(EC_EFPPO_ActorCritic, self).__init__()

        if isinstance(activation, str):
            activation = get_activation(activation)
        # activation is now an nn.Module class (not instance) — call activation() to instantiate
        activation_cls = activation if isinstance(activation, type) else type(activation)

        self.num_actions = int(num_actions)
        self.action_distribution = str(action_distribution).lower()
        if self.action_distribution not in ("gaussian", "beta"):
            raise ValueError(f"Unsupported action_distribution: {action_distribution}")
        actor_output_dim = self.num_actions if self.action_distribution == "gaussian" else 2 * self.num_actions

        # ---- Policy Network (actor) ----
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, hidden_dim))
        actor_layers.append(activation_cls())
        for _ in range(num_hidden_layers - 1):
            actor_layers.append(nn.Linear(hidden_dim, hidden_dim))
            actor_layers.append(activation_cls())
        actor_layers.append(nn.Linear(hidden_dim, actor_output_dim))
        self.actor = nn.Sequential(*actor_layers)

        # ---- Energy Value Network (energy_critic) ----
        energy_critic_layers = []
        energy_critic_layers.append(nn.Linear(num_critic_obs, hidden_dim))
        energy_critic_layers.append(activation_cls())
        for _ in range(num_hidden_layers - 1):
            energy_critic_layers.append(nn.Linear(hidden_dim, hidden_dim))
            energy_critic_layers.append(activation_cls())
        energy_critic_layers.append(nn.Linear(hidden_dim, 1))
        self.energy_critic = nn.Sequential(*energy_critic_layers)

        # ---- Reach Value Network (reach_critic) ----
        reach_critic_layers = []
        reach_critic_layers.append(nn.Linear(num_critic_obs, hidden_dim))
        reach_critic_layers.append(activation_cls())
        for _ in range(num_hidden_layers - 1):
            reach_critic_layers.append(nn.Linear(hidden_dim, hidden_dim))
            reach_critic_layers.append(activation_cls())
        reach_critic_layers.append(nn.Linear(hidden_dim, 1))
        self.reach_critic = nn.Sequential(*reach_critic_layers)

        # ---- Action noise ----
        # 参考 JAX 实现优化 log_std，再通过 exp(log_std) 得到正标准差。
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.bounded_actor_mean = bool(bounded_actor_mean)
        init_noise_std = max(float(init_noise_std), 1e-6)
        init_log_std = torch.log(torch.ones(num_actions) * init_noise_std)
        init_log_std.clamp_(self.log_std_min, self.log_std_max)
        self.log_std = nn.Parameter(init_log_std)
        self.distribution = None
        self.raw_action_mean = None
        self.beta_alpha = None
        self.beta_beta = None
        self._beta_log_range_scale = float(np.log(2.0))
        Normal.set_default_validate_args(False)
        Beta.set_default_validate_args(False)

        # ---- Weight initialization (aligned with JAX version) ----
        self._init_weights()

        print(f"EC_EFPPO Actor: {self.actor}")
        print(f"EC_EFPPO Energy Critic: {self.energy_critic}")
        print(f"EC_EFPPO Reach Critic: {self.reach_critic}")
        print(f"EC_EFPPO action distribution: {self.action_distribution}")
        print(f"EC_EFPPO bounded actor mean: {self.bounded_actor_mean}")

    def _init_weights(self):
        """
        初始化权重，与 JAX 版对齐：
        - 隐藏层: orthogonal(sqrt(2)), bias=0
        - actor 最后一层: orthogonal(0.01)
        - critic 最后一层: orthogonal(1.0)
        """
        # Actor: hidden layers with sqrt(2), last layer with 0.01
        self._init_sequential(self.actor, hidden_gain=np.sqrt(2), output_gain=0.01)

        # Energy critic: hidden layers with sqrt(2), last layer with 1.0
        self._init_sequential(self.energy_critic, hidden_gain=np.sqrt(2), output_gain=1.0)

        # Reach critic: hidden layers with sqrt(2), last layer with 1.0
        self._init_sequential(self.reach_critic, hidden_gain=np.sqrt(2), output_gain=1.0)

    @staticmethod
    def _init_sequential(sequential, hidden_gain, output_gain):
        """对 nn.Sequential 中的 Linear 层应用 orthogonal 初始化。"""
        linear_layers = [m for m in sequential if isinstance(m, nn.Linear)]
        for i, layer in enumerate(linear_layers):
            gain = output_gain if i == len(linear_layers) - 1 else hidden_gain
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    # ---- Distribution helpers ----

    @property
    def std(self):
        """当前动作标准差。由 log_std 限幅后指数映射得到，始终为正。"""
        return torch.exp(torch.clamp(self.log_std, self.log_std_min, self.log_std_max))

    def clamp_log_std_(self):
        """将可学习 log_std 参数保持在配置范围内。"""
        with torch.no_grad():
            self.log_std.clamp_(self.log_std_min, self.log_std_max)

    def load_state_dict(self, state_dict, strict: bool = True):
        """兼容旧 checkpoint：旧版本保存的是实际 std 参数。"""
        if "log_std" not in state_dict and "std" in state_dict:
            state_dict = dict(state_dict)
            old_std = state_dict.pop("std").detach().float().clamp_min(1e-6)
            state_dict["log_std"] = torch.log(old_std).clamp(self.log_std_min, self.log_std_max)
        return super().load_state_dict(state_dict, strict=strict)

    def _bound_action_mean(self, raw_mean):
        """可选地将 actor mean 映射到环境执行动作边界内。"""
        if self.bounded_actor_mean:
            return torch.tanh(raw_mean)
        return raw_mean

    def update_distribution(self, observations):
        actor_output = self.actor(observations)
        if self.action_distribution == "beta":
            alpha_raw, beta_raw = actor_output.split(self.num_actions, dim=-1)
            self.beta_alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
            self.beta_beta = torch.nn.functional.softplus(beta_raw) + 1.0
            self.distribution = Beta(self.beta_alpha, self.beta_beta)
            # Beta policy has no tanh logits. Store bounded mean as raw_action_mean so
            # Gaussian-specific raw mean regularization stays inactive on this branch.
            self.raw_action_mean = self.action_mean
        else:
            self.beta_alpha = None
            self.beta_beta = None
            raw_mean = actor_output
            self.raw_action_mean = raw_mean
            mean = self._bound_action_mean(raw_mean)
            self.distribution = Normal(mean, mean * 0. + self.std)

    def _scale_beta_sample(self, sample):
        return sample * 2.0 - 1.0

    def _unscale_beta_action(self, action):
        return ((action + 1.0) * 0.5).clamp(1e-6, 1.0 - 1e-6)

    def _beta_log_prob(self, actions):
        unscaled = self._unscale_beta_action(actions)
        return (self.distribution.log_prob(unscaled) - self._beta_log_range_scale).sum(dim=-1)

    @property
    def action_mean(self):
        if self.action_distribution == "beta":
            mean01 = self.beta_alpha / (self.beta_alpha + self.beta_beta)
            return self._scale_beta_sample(mean01)
        return self.distribution.mean

    @property
    def action_raw_mean(self):
        return self.raw_action_mean

    @property
    def action_std(self):
        if self.action_distribution == "beta":
            return self.distribution.stddev * 2.0
        return self.distribution.stddev

    @property
    def entropy(self):
        if self.action_distribution == "beta":
            return (self.distribution.entropy() + self._beta_log_range_scale).sum(dim=-1)
        return self.distribution.entropy().sum(dim=-1)

    @property
    def action_dist_alpha(self):
        return self.beta_alpha

    @property
    def action_dist_beta(self):
        return self.beta_beta

    # ---- Core methods ----

    def act(self, observations, critic_observations=None):
        """
        采样动作并计算 energy/reach value。

        Args:
            observations: [N, num_actor_obs] actor 的观测
            critic_observations: [N, num_critic_obs] critic 的观测
                如果为 None，则使用 observations

        Returns:
            action: [N, num_actions] 采样的动作
            log_prob: [N] 动作的 log 概率
            energy_value: [N] energy value function 预测
            reach_value: [N] reach value function 预测
        """
        if critic_observations is None:
            critic_observations = observations

        self.update_distribution(observations)
        if self.action_distribution == "beta":
            action = self._scale_beta_sample(self.distribution.sample())
            log_prob = self._beta_log_prob(action)
        else:
            action = self.distribution.sample()
            log_prob = self.distribution.log_prob(action).sum(dim=-1)

        energy_value = self.energy_critic(critic_observations).squeeze(-1)
        reach_value = self.reach_critic(critic_observations).squeeze(-1)

        return action, log_prob, energy_value, reach_value

    def get_actions_log_prob(self, actions):
        """给定动作，返回 log 概率。"""
        if self.action_distribution == "beta":
            return self._beta_log_prob(actions)
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations):
        """
        仅前向传播两个 critic，用于计算 bootstrap value。

        Args:
            critic_observations: [N, num_critic_obs]

        Returns:
            energy_value: [N] energy value
            reach_value: [N] reach value
        """
        energy_value = self.energy_critic(critic_observations).squeeze(-1)
        reach_value = self.reach_critic(critic_observations).squeeze(-1)
        return energy_value, reach_value

    def act_inference(self, observations):
        """
        确定性推理（用均值而非采样），用于部署和评估。

        Args:
            observations: [N, num_actor_obs]

        Returns:
            actions_mean: [N, num_actions]
        """
        actor_output = self.actor(observations)
        if self.action_distribution == "beta":
            alpha_raw, beta_raw = actor_output.split(self.num_actions, dim=-1)
            alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
            beta = torch.nn.functional.softplus(beta_raw) + 1.0
            return self._scale_beta_sample(alpha / (alpha + beta))
        return self._bound_action_mean(actor_output)
