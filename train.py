
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from Agent import Agent
from Env import Env
from arm import Viewer
from utils import sigmoid
from Memory import ReplayBuffer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# seed
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


episodes = 100000
max_step = 200
reward_scale = 1
gamma = 0.99

batch_size = 2 ** 9 # 256
repeat_times = 2 ** 3 # 16
target_step = 2 ** 12 # 2 ** 12
max_memo = target_step

env = Env()
agent = Agent()
buffer = ReplayBuffer(max_len=max_memo + max_step, state_dim=20, action_dim=4)

def train():
    sum_reward = []
    viewer = Viewer()
    for e in range(episodes):
        buffer.empty_buffer_before_explore()
        actual_step = 0
        while actual_step < target_step: # batch_size
            state, goal = env.reset(viewer)
            tmp_reward = 0
            step_sum = 0
            for i in range(max_step):
                step_sum += 1
                with torch.no_grad():
                    action, noise = agent.actor.get_action_noise(torch.FloatTensor(state).cuda())
                    next_state, reward, done = env.step(action, goal, viewer)
                    actual_step += 1
                    # if e % 5 == 0:
                    #     env.renders(viewer,
                    #         next_state[15],
                    #         next_state[16],
                    #         next_state[17],
                    #         next_state[18],
                    #         goal)
                    tmp_reward += reward

                    other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
                    buffer.append_buffer(state, other)
                    # print("done: ", done)
                    if done:
                        break
                    state = next_state
            print("sum_step: ", step_sum)
            sum_reward.append(tmp_reward)
        if e % 50 == 0:
            torch.save(agent.actor.state_dict(), "./model_test1/modela{}_.pth".format(e))
            torch.save(agent.critic.state_dict(), "./model_test1/modelc{}_.pth".format(e))
            print('model saved')
        obj_a, obj_c = agent.update_net(buffer, target_step, batch_size, repeat_times)

        if e % 10 == 0:
            plt.plot(sum_reward)
            plt.pause(1)
            plt.close()

        print("epidode: ", e)
        print("obj_a: ", obj_a)
        print("obj_c: ", obj_c)
        print("============")

    viewer.end_viewer()

if __name__ == "__main__":
    train()

