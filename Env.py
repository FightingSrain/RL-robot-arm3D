import numpy as np
from arm import Viewer
from utils import sigmoid


class Env():
    def __init__(self):
        self.state = None
        self.pre_state = None
        # self.viewer = Viewer()
        self.arm_angle = [90, 90, 90, 90]

    def reset(self, viewer):
        # 初始化随机角度
        start_t1 = np.random.rand() * np.pi
        start_t2 = np.random.rand() * np.pi
        start_t3 = np.random.rand() * np.pi
        start_t4 = np.random.rand() * np.pi
        self.state, goal = viewer.reset(start_t1, start_t2, start_t3, start_t4)  # [0, 180]
        self.on_goal = 0
        return self.state, goal

    def renders(self, viewer, angle1, angle2, angle3, angle4, goal):
        viewer.render(angle1, angle2, angle3, angle4, goal)

    def step(self, action, goal, viewer):
        self.pre_state = self.state.copy()
        new_t1 = np.clip(self.pre_state[15] + np.tanh(action[0]) * 0.1, a_min=0., a_max=np.pi)
        new_t2 = np.clip(self.pre_state[16] + np.tanh(action[1]) * 0.1, a_min=0., a_max=np.pi)
        new_t3 = np.clip(self.pre_state[17] + np.tanh(action[2]) * 0.1, a_min=0., a_max=np.pi)
        new_t4 = np.clip(self.pre_state[18] + np.tanh(action[3]) * 0.1, a_min=0., a_max=np.pi)

        # 会出现抖动的情况
        # new_t1 = (self.pre_state[15] + np.tanh(action[0]) * 0.1) % np.pi
        # new_t2 = (self.pre_state[16] + np.tanh(action[1]) * 0.1) % np.pi
        # new_t3 = (self.pre_state[17] + np.tanh(action[2]) * 0.1) % np.pi
        # new_t4 = (self.pre_state[18] + np.tanh(action[3]) * 0.1) % np.pi
        coord = viewer.cal_arm(new_t1, new_t2, new_t3, new_t4)  # 计算动作后的坐标
        dist1 = np.sqrt((coord[0][0] - goal[0]) ** 2 + (coord[1][0] - goal[1]) ** 2 + (coord[2][0] - goal[2]) ** 2)
        dist2 = np.sqrt((coord[0][1] - goal[0]) ** 2 + (coord[1][1] - goal[1]) ** 2 + (coord[2][1] - goal[2]) ** 2)
        dist3 = np.sqrt((coord[0][2] - goal[0]) ** 2 + (coord[1][2] - goal[1]) ** 2 + (coord[2][2] - goal[2]) ** 2)
        # masks = 0 if (dist3 > 1) else 1

        # r1 = np.sqrt((coord[0][2] - goal[0]) ** 2 + (coord[1][2] - goal[1]) ** 2 + (coord[2][2] - goal[2]) ** 2)
        # r2 = np.sqrt((self.pre_state[2] - goal[0]) ** 2 +
        #              (self.pre_state[5] - goal[1]) ** 2 +
        #              (self.pre_state[8] - goal[2]) ** 2)
        #
        # reward = (r2 - r1)*30
        # if masks == 1:
        #     print(masks)
        reward = -dist3
        done = False
        if dist3 < 2/30:
            reward += 1
            self.on_goal += 1
            done = (self.on_goal >= 1)
        else:
            self.on_goal = 0
        mask = 1. if self.on_goal else 0.

        # 距离用/30, 坐标用/30
        self.state = np.asarray([coord[0][0], coord[0][1], coord[0][2],
                                 coord[1][0], coord[1][1], coord[1][2],
                                 coord[2][0], coord[2][1], coord[2][2], # 6
                                 dist1, dist2, dist3,
                                 goal[0], goal[1], goal[2],
                                 new_t1, new_t2, new_t3, new_t4,  # 15
                                 mask])

        return self.state, reward, done
