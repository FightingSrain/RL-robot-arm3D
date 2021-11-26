
import torch
import numpy as np
import numpy.random as rd

class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = action_dim  # for self.sample_for_ppo(

        self.if_gpu = False
        other_dim = 1 + 1 + action_dim * 2
        self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
        self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)

    def append_buffer(self, state, other):  # CPU array to CPU array
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_buffer(self, state, other):  # CPU array to CPU array
        size = len(other)
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            if next_idx > self.max_len:
                self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
                self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True
            next_idx = next_idx - self.max_len

            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def sample_batch(self, batch_size):
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],  # reward
                r_m_a[:, 1:2],  # mask = 0.0 if done else gamma
                r_m_a[:, 2:],  # action
                self.buf_state[indices],  # state
                self.buf_state[indices + 1])  # next_state

    def sample_for_ppo(self):
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (all_other[:, 0],  # reward
                all_other[:, 1],  # mask = 0.0 if done else gamma
                all_other[:, 2:2 + self.action_dim],  # action
                all_other[:, 2 + self.action_dim:],  # noise
                torch.as_tensor(self.buf_state[:self.now_len], device=self.device))  # state

    def update_now_len_before_sample(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer_before_explore(self):
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False

# batch_size = 2 ** 8 # 256
# repeat_times = 2 ** 4 # 16
# target_step = 2 ** 12 # 2 ** 12
# max_memo = target_step
# buffer = ReplayBuffer(max_len=max_memo + 200, state_dim=[3, 64, 64], action_dim=[1, 64, 64])
# reward_scale = 1
# gamma = 0.99
# for i in range(1024):
#     state = np.ones((1, 3, 64, 64))
#     reward = np.ones((1, 1, 64, 64))
#     action = np.ones((1, 1, 64, 64))
#     noise = np.ones((1, 1, 64, 64))
#     done = True
#     other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
#     buffer.append_buffer(state, other)
#
#
#
#
# buf_reward, buf_mask, buf_action, buf_noise, buf_state = buffer.sample_for_ppo()
# print(buf_reward.shape)
# print(buf_mask.shape)
# print(buf_action.shape)
# print(buf_noise.shape)
# print(buf_state.shape)
# print("===========")


