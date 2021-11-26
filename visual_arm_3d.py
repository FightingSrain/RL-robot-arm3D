import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from Agent import Agent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# env_tst = Env()
agent_tst = Agent()
agent_tst.actor.load_state_dict(torch.load("./model_test1/modela1750_.pth"))


def cal_arm(t1, t2, t3, t4):
    # parameter
    arm_len = 10
    # angle = [0, 180]
    theta1 = t1
    theta2 = t2
    theta3 = t3
    theta4 = t4

    e1 = np.cos(theta1) * arm_len
    x1 = np.cos(theta4) * e1
    y1 = np.sin(theta4) * e1
    z1 = np.sin(theta1) * arm_len

    tmp1 = theta1 - np.pi / 2 + theta2
    e2 = np.cos(tmp1) * arm_len
    x2 = np.cos(theta4) * e2 + x1
    y2 = np.sin(theta4) * e2 + y1
    z2 = np.sin(tmp1) * arm_len + z1

    tmp2 = theta3 - np.pi / 2 + tmp1
    e3 = np.cos(tmp2) * arm_len
    x3 = np.cos(theta4) * e3 + x2
    y3 = np.sin(theta4) * e3 + y2
    z3 = np.sin(tmp2) * arm_len + z2
    return np.asarray([[x1, x2, x3],
                       [y1, y2, y3],
                       [z1, z2, z3]]) / 30


def start_state(gx, gy, gz):
    flag = 0
    if gx<0 and gy>=0:
        gx = np.abs(gx)
        flag = 1
    elif gx<0 and gy<0:
        gx = np.abs(gx)
        gy = np.abs(gy)
        flag = 2
    elif gx>=0 and gy<0:
        gy = np.abs(gy)
        flag = 3
    coord = np.asarray([[0., 0., 0.],
                        [0., 0., 0.],
                        [10, 20, 30]]) / 30
    dist1 = np.sqrt((coord[0][0] - gx / 30) ** 2 + (coord[1][0] - gy / 30) ** 2 + (coord[2][0] - gz / 30) ** 2)
    dist2 = np.sqrt((coord[0][1] - gx / 30) ** 2 + (coord[1][1] - gy / 30) ** 2 + (coord[2][1] - gz / 30) ** 2)
    dist3 = np.sqrt((coord[0][2] - gx / 30) ** 2 + (coord[1][2] - gy / 30) ** 2 + (coord[2][2] - gz / 30) ** 2)
    mask = 1. if (dist3 < 2 / 30) else 0.
    state = np.asarray([coord[0][0], coord[0][1], coord[0][2],
                        coord[1][0], coord[1][1], coord[1][2],
                        coord[2][0], coord[2][1], coord[2][2],
                        dist1, dist2, dist3,
                        gx/30, gy/30, gz/30, # 12
                        np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2,  # 15
                        mask, flag])
    return state


def mid_state(state, _coordx, _coordy, _coordz):
    flag = 0
    if _coordx[0][-1] < 0 and _coordy[0][-1] >= 0:
        _coordx[0][-1] = -_coordx[0][-1]
        flag = 1
    elif _coordx[0][-1] < 0 and _coordy[0][-1] < 0:
        _coordx[0][-1] = -_coordx[0][-1]
        _coordy[0][-1] = -_coordy[0][-1]
        flag = 2
    elif _coordx[0][-1] >= 0 and _coordy[0][-1] < 0:
        _coordy[0][-1] = -_coordy[0][-1]
        flag = 3
    dist1 = np.sqrt(
        (state[0] - _coordx[0][-1] / 30) ** 2 + (state[3] - _coordy[0][-1] / 30) ** 2 + (
                    state[6] - _coordz[0][-1] / 30) ** 2)
    dist2 = np.sqrt(
        (state[1] - _coordx[0][-1] / 30) ** 2 + (state[4] - _coordy[0][-1] / 30) ** 2 + (
                    state[7] - _coordz[0][-1] / 30) ** 2)
    dist3 = np.sqrt(
        (state[2] - _coordx[0][-1] / 30) ** 2 + (state[5] - _coordy[0][-1] / 30) ** 2 + (
                    state[8] - _coordz[0][-1] / 30) ** 2)
    masks = 0. if (dist3 > 2 / 30) else 1.
    state[9] = dist1
    state[10] = dist2
    state[11] = dist3
    state[12] = _coordx[0][-1] / 30
    state[13] = _coordy[0][-1] / 30
    state[14] = _coordz[0][-1] / 30
    state[19] = masks
    state[-1] = flag

    return state


def step(state, action):
    pre_state = state.copy()
    new_t1 = np.clip(pre_state[15] + np.tanh(action[0]) * 0.1, a_min=0., a_max=np.pi)
    new_t2 = np.clip(pre_state[16] + np.tanh(action[1]) * 0.1, a_min=0., a_max=np.pi)
    new_t3 = np.clip(pre_state[17] + np.tanh(action[2]) * 0.1, a_min=0., a_max=np.pi)
    new_t4 = np.clip(pre_state[18] + np.tanh(action[3]) * 0.1, a_min=0., a_max=np.pi)
    coord_ = cal_arm(new_t1, new_t2, new_t3, new_t4)  # 计算动作后的坐标
    dist1_ = np.sqrt((coord_[0][0] - pre_state[12]) ** 2 + (coord_[1][0] - pre_state[13]) ** 2 +
                     (coord_[2][0] - pre_state[14]) ** 2)
    dist2_ = np.sqrt((coord_[0][1] - pre_state[12]) ** 2 + (coord_[1][1] - pre_state[13]) ** 2 +
                     (coord_[2][1] - pre_state[14]) ** 2)
    dist3_ = np.sqrt((coord_[0][2] - pre_state[12]) ** 2 + (coord_[1][2] - pre_state[13]) ** 2 +
                     (coord_[2][2] - pre_state[14]) ** 2)
    mask = 1. if dist3_ < 2 / 30 else 0.
    done = True if dist3_ < 2 / 30 else False
    # 距离用/60, 坐标用/30
    next_state = np.asarray([coord_[0][0], coord_[0][1], coord_[0][2],
                             coord_[1][0], coord_[1][1], coord_[1][2],
                             coord_[2][0], coord_[2][1], coord_[2][2],
                             dist1_, dist2_, dist3_,
                             pre_state[12], pre_state[13], pre_state[14],
                             new_t1, new_t2, new_t3, new_t4,  # 15
                             mask, pre_state[-1]])
    return next_state, coord_, done

def draw(ax, next_state, coord_):
    # print(next_state[0:3])
    if next_state[-1] == 1.0: #
        # coord_[0][:] *= -1
        next_state[0:3] *= -1
        next_state[12] *= -1
    elif next_state[-1] == 2.0: #
        next_state[0:3] *= -1
        next_state[3:6] *=-1
        next_state[12] *= -1
        next_state[13] *= -1
        # coord_[1][:] *= -1
    elif next_state[-2] == 3.0: #
        # coord_[1][:] *= -1
        next_state[3:6] *= -1
        next_state[13] *= -1
    # print(next_state[-1])
    # print(next_state[12:15])
    # print(next_state[0:3])
    # print("000000")
    # 画图
    x = np.around([[0, next_state[0] * 30, next_state[1] * 30, next_state[2] * 30]], 5)  # 要连接的两个点的坐标
    y = np.around([[0, next_state[3] * 30, next_state[4] * 30, next_state[5] * 30]], 5)
    z = np.around([[0, next_state[6] * 30, next_state[7] * 30, next_state[8] * 30]], 5)
    xp = np.linspace(0, 30, 30)
    yp = np.linspace(0, 30, 30)
    X, Y = np.meshgrid(xp, yp)
    ax.plot_surface(X,
                    Y,
                    Z=X * 0 + 0,
                    color='g',
                    alpha=0.6
                    )
    for i in range(len(x)):
        ax.plot(x[i], y[i], z[i], c='r')
        ax.scatter(x[i], y[i], z[i], c='b', marker=".")
        # plt.gca().set_aspect(1)
    ax.scatter(0, 0, 0, s=80, c='b', marker=".")
    ax.scatter(next_state[2] * 30, next_state[5] * 30, next_state[8] * 30, s=50, c='g', marker="o")
    ax.scatter(next_state[12] * 30, next_state[13] * 30, next_state[14] * 30, s=50, c='b', marker="x")
    # 设置坐标轴图标
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    # 设置坐标轴范围
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_zlim(-5, 35)

def Loop(states, plt, ax):
    state = states.copy()
    done_state = None
    for i in range(200):
        plt.cla()
        action = agent_tst.actor(torch.FloatTensor(state[:-1]).cuda()).detach().cpu().numpy()
        # ==============
        next_state, coord_, done = step(state, action)
        # print(next_state[12:14])
        # print("llllllll")
        draw(ax, next_state.copy(), coord_)
        # 暂停
        plt.pause(0.01)
        if done:
            done_state = next_state
            break
        done_state = next_state
        state = next_state
    return done_state

