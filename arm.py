import numpy as np
import matplotlib.pyplot as plt



class Viewer():
    def __init__(self):
        # 生成画布
        self.fig = plt.figure()
        # 打开交互模式
        plt.ion()
    def cal_arm(self, t1, t2, t3, t4):
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

        tmp1 = theta1 - np.pi/2 + theta2
        e2 = np.cos(tmp1) * arm_len
        x2 = np.cos(theta4) * e2 + x1
        y2 = np.sin(theta4) * e2 + y1
        z2 = np.sin(tmp1) * arm_len + z1

        tmp2 = theta3 - np.pi/2 + tmp1
        e3 = np.cos(tmp2) * arm_len
        x3 = np.cos(theta4) * e3 + x2
        y3 = np.sin(theta4) * e3 + y2
        z3 = np.sin(tmp2) * arm_len + z2
        return np.asarray([[x1, x2, x3],
                [y1, y2, y3],
                [z1, z2, z3]])/30
    def on_draw(self, t1, t2, t3, t4, goal):
        # 清除原有图像
        self.fig.clf()
        # 设定标题等
        self.fig.suptitle("machanical_arm_3D")

        coord = self.cal_arm(t1, t2, t3, t4)
        x = np.around([[0, coord[0][0]*30, coord[0][1]*30, coord[0][2]*30]], 5)  # 要连接的两个点的坐标
        y = np.around([[0, coord[1][0]*30, coord[1][1]*30, coord[1][2]*30]], 5)
        z = np.around([[0, coord[2][0]*30, coord[2][1]*30, coord[2][2]*30]], 5)

        # 生成画布
        ax = self.fig.add_subplot(111, projection="3d")
        xp = np.linspace(-30, 30, 60)
        yp = np.linspace(-30, 30, 60)
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
        ax.scatter(coord[0][2]*30, coord[1][2]*30, coord[2][2]*30, s=50, c='g', marker="o")
        ax.scatter(goal[0]*30, goal[1]*30, goal[2]*30, s=50, c='b', marker="x")
        # 设置坐标轴图标
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")

        # 设置坐标轴范围
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_zlim(-5, 35)
        # 暂停
        plt.pause(0.01)

    def render(self, t1, t2, t3, t4, goal):
        self.on_draw(t1, t2, t3, t4, goal)

    def end_viewer(self):
        # 关闭交互模式
        plt.ioff()
        # 图形显示
        plt.show()
    def reset(self, t1, t2, t3, t4):
        coord = self.cal_arm(t1, t2, t3, t4)
        goalx = np.random.rand()
        goaly = np.random.rand()
        goalz = np.random.rand()
        # goalx = 10
        # goaly = 10
        # goalz = 10
        goal = np.asarray([goalx, goaly, goalz])  # 随机目标点
        dist1 = np.sqrt((coord[0][0] - goal[0]) ** 2 + (coord[1][0] - goal[1]) ** 2 + (coord[2][0] - goal[2]) ** 2)
        dist2 = np.sqrt((coord[0][1] - goal[0]) ** 2 + (coord[1][1] - goal[1]) ** 2 + (coord[2][1] - goal[2]) ** 2)
        dist3 = np.sqrt((coord[0][2] - goal[0]) ** 2 + (coord[1][2] - goal[1]) ** 2 + (coord[2][2] - goal[2]) ** 2)
        mask = 0. if (dist3 > 2/30) else 1
        state = [coord[0][0], coord[0][1], coord[0][2],
                 coord[1][0], coord[1][1], coord[1][2],
                 coord[2][0], coord[2][1], coord[2][2],
                 dist1, dist2, dist3,
                 goalx, goaly, goalz,
                 t1, t2, t3, t4, # 15, 16, 17, 18
                 mask] # /60 normlization
        # state = np.around(state, 5)
        return state, goal

