

import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from RL_model_3d.Agent import Agent
from RL_model_3d.Env import Env
from RL_model_3d.arm import Viewer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

env_tst = Env()
agent_tst = Agent()
agent_tst.actor.load_state_dict(torch.load("./model_test1/modela1750_.pth"))
def arm_tst():
    viewer = Viewer()
    state, goal = env_tst.reset(viewer)

    state[12] = 20 / 30  # x
    state[13] = 20 / 30  # y
    state[14] = 0 / 30  # z
    goal = np.asarray([state[12], state[13], state[14]])
    tmp_reward = 0
    step_sum = 0
    for i in range(200):
        step_sum += 1
        with torch.no_grad():
            action = agent_tst.actor(torch.FloatTensor(state).cuda()).detach().cpu().numpy()
            next_state, reward, done = env_tst.step(action, goal, viewer)
            # print(next_state[11])
            print(reward)
            env_tst.renders(viewer,
                next_state[15],
                next_state[16],
                next_state[17],
                next_state[18],
                goal)
            tmp_reward += reward
            if done:
                break
            state = next_state
    print("step_num: ", step_sum)
if __name__ == "__main__":
    arm_tst()