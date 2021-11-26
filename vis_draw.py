import cv2
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.backend_bases import MouseEvent
import numpy as np

from arm import Viewer
from visual_arm_3d import Loop, start_state, mid_state
class DrawPoint:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 12), dpi=80)

        self.fig.suptitle("machanical_arm_3D")
        self.ax_1 = self.fig.add_subplot(221)
        self.ax_3d = self.fig.add_subplot(222, projection="3d")
        self.ax_2 = self.fig.add_subplot(223)
        self.ax_3 = self.fig.add_subplot(224, projection="3d")
        # 分别设置坐标轴和坐标轴范围
        self.ax_1.set_xlabel("X Label")
        self.ax_1.set_ylabel("Y Label")
        self.ax_1.set_xlim(0, 30)
        self.ax_1.set_ylim(0, 30)
        self.ax_1.grid()
        # ------------
        self.ax_2.set_xlabel("X Label")
        self.ax_2.set_ylabel("Z Label")
        self.ax_2.set_xlim(0, 30)
        self.ax_2.set_ylim(0, 30)
        self.ax_2.grid()
        # ------------
        self.ax_3d.set_xlabel("X Label")
        self.ax_3d.set_ylabel("Y Label")
        self.ax_3d.set_zlabel("Z Label")
        self.ax_3d.set_xlim(0, 30)
        self.ax_3d.set_ylim(0, 30)
        self.ax_3d.set_zlim(-5, 35)
        #-------------
        self.ax_3.set_xlabel("X Label")
        self.ax_3.set_ylabel("Y Label")
        self.ax_3.set_zlabel("Z Label")
        self.ax_3.set_xlim(0, 30)
        self.ax_3.set_ylim(0, 30)
        self.ax_3.set_zlim(-5, 35)
        #=============
        self.coord = [[], [], []] # 记录所有轨迹点
        # 记录图中显示节点（目标点）
        self._coordx = [[]]
        self._coordy = [[]]
        self._coordz = [[]]
        self.press = False
        self.state = None

    def on_press(self, event):
        if event.inaxes:  # 判断鼠标是否在axes内
            if event.button == MouseButton.RIGHT:  # 判断按下的是否为鼠标左键
                # print("Start drawing")
                self.press = True

    def on_move(self, event):
        if event.inaxes:
            # coord = [0, 0, 0]
            if self.press:
                # 获取鼠标像素坐标值
                w = event.x
                q = event.y
                # l1 = self.ax.transData.inverted().transform((w, q)) # 像素坐标转化为数据坐标
                l1 = self.ax_1.transAxes.inverted().transform((w, q)) #
                # x, y = event.inaxes.transData.inverted().transform((event.x, event.y))
                if l1[0]>0 and l1[0]<1 and l1[1]>0 and l1[1]<1:
                    x = event.xdata
                    y = event.ydata
                    # tmp1 = self.ax_1.plot(x, y, '.', c='r')  # 画点
                    self.coord[0].append(x)
                    self.coord[1].append(y)
                    # 补齐坐标
                    if len(self.coord[2]) != 0:
                        self.coord[2].append(self.coord[2][-1])
                    else:
                        self.coord[2].append(0)
                elif l1[0]>0 and l1[0]<1 and l1[1]>-1.2 and l1[1]<-0.2:
                    x = event.xdata
                    z = event.ydata
                    # tmp2 = self.ax_2.plot(x, z, '.', c='r')  # 画点
                    self.coord[0].append(x)
                    self.coord[2].append(z)
                    # 补齐坐标
                    if len(self.coord[1]) != 0:
                        self.coord[1].append(self.coord[1][-1])
                    else:
                        self.coord[1].append(0)

    def on_release(self, event):
        if self.press:
            self.press = False  # 鼠标松开,开始绘制
            self._coordx[0].append(self.coord[0][-1])
            self._coordy[0].append(self.coord[1][-1])
            self._coordz[0].append(self.coord[2][-1])
            print(self._coordx)
            print(self._coordy)
            print(self._coordz)
            print("-------")
            for i in range(len(self._coordx)):
                self.ax_1.plot(self._coordx[i], self._coordy[i],  c='r')  # 画线
                self.ax_2.plot(self._coordx[i], self._coordz[i],  c='r')  # 画线
                self.ax_1.plot(self._coordx[i], self._coordy[i], '.', c='g')  # 画点
                self.ax_2.plot(self._coordx[i], self._coordz[i], '.', c='g')  # 画点
                self.ax_3d.plot(self._coordx[i], self._coordy[i], self._coordz[-1], c='r')
                self.ax_3d.scatter(self._coordx[i], self._coordy[i], self._coordz[-1], '.', c='g')
            # 删除 除最后一个点之外的其它点
            del self.coord[0][:-1]
            del self.coord[1][:-1]
            del self.coord[2][:-1]
            self.fig.canvas.draw()  # 更新画布
            #---------------
            plt.ion()

            if len(self._coordx[0]) <= 1: # 初始状态
                self.state = start_state(self._coordx[0][-1], self._coordy[0][-1],
                     self._coordz[0][-1])
                self.state = Loop(self.state, plt, self.ax_3)
            else:
                self.state = mid_state(self.state, self._coordx, self._coordy, self._coordz)
                dist1 = np.sqrt(
                    (self.state[0] - self._coordx[0][-1]/30) ** 2 + (self.state[3] - self._coordy[0][-1]/30) ** 2 + (self.state[6] - self._coordz[0][-1]/30) ** 2)
                dist2 = np.sqrt(
                    (self.state[1] - self._coordx[0][-1]/30) ** 2 + (self.state[4] - self._coordy[0][-1]/30) ** 2 + (self.state[7] - self._coordz[0][-1]/30) ** 2)
                dist3 = np.sqrt(
                    (self.state[2] - self._coordx[0][-1]/30) ** 2 + (self.state[5] - self._coordy[0][-1]/30) ** 2 + (self.state[8] - self._coordz[0][-1]/30) ** 2)
                masks = 0. if (dist3 > 2/30) else 1.
                self.state[9] = dist1
                self.state[10] = dist2
                self.state[11] = dist3
                self.state[12] = self._coordx[0][-1]/30
                self.state[13] = self._coordy[0][-1]/30
                self.state[14] = self._coordz[0][-1]/30
                self.state[19] = masks

                self.state = Loop(self.state, plt, self.ax_3)

            plt.ioff()  # 关闭交互模式
            # 图形显示
            # plt.show()

    def connect(self):
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)


if __name__ == "__main__":
    draw = DrawPoint()
    draw.connect()  # 启动callback
    plt.show()
