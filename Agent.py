
import math

import torch
from torch.optim import *
from torch import FloatTensor, LongTensor

import numpy as np
from net import ActorPPO, CriticAdv


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:

    def __init__(self):

        self.gamma = 0.99
        self.lamda = 0.98 # lambda 值越大 方差大，偏差小
        self.clip_now = 0.25
        self.sqrt_2pi_log = 0.9189385332046727
        self.learning_rate = 0.0001
        self.device = device
        # ============
        self.criterion = torch.nn.SmoothL1Loss()
        # ============
        self.actor = ActorPPO(128, 20, 4).to(device)
        self.critic = CriticAdv(20, 128).to(device)
        # self.actor.load_state_dict(torch.load("./model_test1/modela1100_.pth"))
        # self.critic.load_state_dict(torch.load("./model_test1/modelc1100_.pth"))
        # ============
        self.optimizer = torch.optim.Adam([{'params': self.actor.parameters(), 'lr': self.learning_rate},
                                           {'params': self.critic.parameters(), 'lr': self.learning_rate}])
        # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

    def compute_reward_adv(self, max_memo, buf_reward, buf_mask, buf_value):
        buf_r_sum = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_sum = 0  # reward sum of previous step
        for i in range(max_memo - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value.squeeze(1))
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

    def compute_reward_gae(self, max_memo, buf_reward, buf_mask, buf_value):
        buf_r_sum = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0  # reward sum of previous step
        pre_advantage = 0  # advantage value of previous step
        for i in range(max_memo - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_advantage[i] = buf_reward[i] + buf_mask[i] * pre_advantage - buf_value[i]
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lamda

        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage



    def update_net(self, buffer, _target_step, batch_size, repeat_times=8):
        buffer.update_now_len_before_sample()
        max_memo = buffer.now_len  # assert max_memo >= _target_step

        with torch.no_grad():  # Trajectory using reverse reward
            buf_reward, buf_mask, buf_action, buf_noise, buf_state = buffer.sample_for_ppo()
            # 2 ** 10 **
            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.critic(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_logprob = -(buf_noise.pow(2).__mul__(0.5) + self.actor.a_std_log + self.actor.sqrt_2pi_log).sum(1)
            buf_r_sum, buf_advantage = self.compute_reward_gae(max_memo, buf_reward, buf_mask, buf_value)
            del buf_reward, buf_mask, buf_noise

        obj_critic = None
        for _ in range(int(repeat_times * max_memo / batch_size)):  # PPO: Surrogate objective of Trust Region
            indices = torch.randint(max_memo, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob = self.actor.compute_logprob(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.clip_now, 1 + self.clip_now)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * 0.01

            value = self.critic(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)

            obj_united = obj_actor + obj_critic / (r_sum.std() + 1e-5)
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()
            # self.scheduler.step()
        return self.actor.a_std_log.mean().item(), obj_critic.item()













