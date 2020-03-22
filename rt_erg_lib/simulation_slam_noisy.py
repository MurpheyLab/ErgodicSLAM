import matplotlib.pyplot as plt
from matplotlib import animation
import autograd.numpy as np
from .utils import convert_ck2dist, convert_traj2ck
from tqdm import tqdm
import time
import numpy as np
from numpy import sin, cos, sqrt

class simulation_slam():
    def __init__(self, size, init_state, t_dist, model_true, erg_ctrl_true, env_true, model_dr, erg_ctrl_dr, env_dr, tf, landmarks, sensor_range):
        self.size = size
        self.init_state = init_state
        self.tf = tf
        self.t_dist = t_dist
        self.exec_times = np.zeros(tf)
        self.landmarks = landmarks
        self.sensor_range = sensor_range

        self.erg_ctrl_true = erg_ctrl_true
        self.env_true = env_true
        self.model_true = model_true

        self.erg_ctrl_dr = erg_ctrl_dr
        self.env_dr = env_dr
        self.model_dr = model_dr

    def start(self, noise=0., report=False):
        # initialization
        self.log = {'trajectory_true': [], 'trajectory_dr': [], 'observation': []}
        state_true = self.env_true.reset(self.init_state)
        state_dr = self.env_dr.reset(self.init_state)

        # simulation loop
        for t in tqdm(range(self.tf)):
            start_time = time.time()

            # this is what robot thinks it's doing
            ctrl_dr = self.erg_ctrl_dr(state_true)
            state_dr = self.env_dr.step(ctrl_dr)
            self.log['trajectory_dr'].append(state_dr)

            # this is what robot is actually doing (if we don't correct it)
            ctrl_true = ctrl_dr + noise * np.random.randn(self.env_true.action_space.shape[0])
            state_true = self.env_true.step(ctrl_true)
            self.log['trajectory_true'].append(state_true)

            # observation model
            # obs = []
            # for i in range(self.landmarks.shape[0]):
            #
            #     item = self.landmarks[i]
            #     dist = sqrt((item[0]-state[0])**2 + (item[1]-state[1])**2)
            #     if(dist <= self.sensor_range):
            #         obs.append(i)
            # self.log['observation'].append(obs)

            self.exec_times[t] = time.time()-start_time
        print("simulation finished.")
        if report:
            print(self.exec_times[1:10])
            print("mean execution time: {0:.6f}(s), standard deviation: {1:.6f}(s)".format(np.mean(self.exec_times), np.std(self.exec_times)))

    def plot(self, point_size=1, save=None):
        [xy, vals] = self.t_dist.get_grid_spec()
        plt.contourf(*xy, vals, levels=20)

        xt_true = np.stack(self.log['trajectory_true'])
        traj_true = plt.scatter(xt_true[:self.tf, 0], xt_true[:self.tf, 1], s=point_size, c='red')
        xt_dr = np.stack(self.log['trajectory_dr'])
        traj_dr = plt.scatter(xt_dr[:self.tf, 0], xt_dr[:self.tf, 1], s=point_size, c='cyan')

        plt.legend([traj_true, traj_dr], ['Noise-Free Path', 'True Path'])

        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        if save is not None:
            plt.savefig(save)
        plt.show()
        # return plt.gcf()

    def animate(self, point_size=1, show_label=False, show_traj=True, save=None, rate=50):
        [xy, vals] = self.t_dist.get_grid_spec()
        plt.contourf(*xy, vals, levels=20)
        plt.scatter(self.landmarks[:,0], self.landmarks[:,1], color='white')
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        fig = plt.gcf()

        xt_true = np.stack(self.log['trajectory_true'])
        points_true = ax.scatter([], [], s=point_size, c='red')
        xt_dr = np.stack(self.log['trajectory_dr'])
        points_dr = ax.scatter([], [], s=point_size, c='cyan')

        plt.legend([points_true, points_dr], ['Noise-Free Path', 'True Path'])

        sensor_points = []
        for id in range(self.landmarks.shape[0]):
            sensor_point = ax.plot([], [], color='orange')
            sensor_points.append(sensor_point)

        def sub_animate(i):
            # print("iter: ", i)

            if(show_traj):
                points_true.set_offsets(np.array([xt_true[:i, 0], xt_true[:i, 1]]).T)
                points_dr.set_offsets(np.array([xt_dr[:i, 0], xt_dr[:i, 1]]).T)
            else:
                points_true.set_offsets(np.array([[xt_true[i, 0]], [xt_true[i, 1]]]).T)
                points_true.set_offsets(np.array([[xt_dr[i, 0]], [xt_dr[i, 1]]]).T)

            # observation model visualization
            # for id in range(self.landmarks.shape[0]):
            #     if id in self.log['observation'][i]:
            #         sensor_points[id][0].set_xdata([xt[i, 0], self.landmarks[id][0]])
            #         sensor_points[id][0].set_ydata([xt[i, 1], self.landmarks[id][1]])
            #     else:
            #         sensor_points[id][0].set_xdata([])
            #         sensor_points[id][0].set_ydata([])

            if show_label:
                cx_true = round(xt_true[i, 0], 2)
                cy_true = round(xt_true[i, 1], 2)
                cth_true = round(xt_true[i, 2], 2)
                quiver_true = ax.quiver(cx_true, cy_true, cos(cth_true), sin(cth_true), color='red')
                cx_dr = round(xt_dr[i, 0], 2)
                cy_dr = round(xt_dr[i, 1], 2)
                cth_dr = round(xt_dr[i, 2], 2)
                quiver_dr = ax.quiver(cx_dr, cy_dr, cos(cth_dr), sin(cth_dr), color='cyan')
                if i == self.tf-1:
                    quiver_true.remove()
                    quiver_dr.remove()
                    # for id in range(self.landmarks.shape[0]):
                    #     sensor_points[id][0].set_xdata([])
                    #     sensor_points[id][0].set_ydata([])
                    points_true.set_offsets(np.array([[xt_true[i, 0]], [xt_true[i, 1]]]).T)
                    points_dr.set_offsets(np.array([[xt_dr[i, 0]], [xt_dr[i, 1]]]).T)
                    ret = [points_true, points_dr]
                else:
                    ret = [points_true, quiver_true, points_dr, quiver_dr]
                # for item in sensor_points:
                #     ret.append(item[0])
                return ret
            else:
                ret = [points_true, points_dr]
                # for item in sensor_points:
                #     ret.append(item[0])
                return ret

        anim = animation.FuncAnimation(fig, sub_animate, frames=self.tf, interval=(1000/rate), blit=True)
        if save is not None:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=40, metadata=dict(artist='simulation_slam'), bitrate=5000)
            anim.save(save, writer=writer)
        plt.show()
        # return anim

    def path_reconstruct(self, save=None):
        xy, vals = self.t_dist.get_grid_spec()

        path_true = np.stack(self.log['trajectory_true'])[:self.tf, self.model_true.explr_idx]
        ck_true = convert_traj2ck(self.erg_ctrl_true.basis, path_true)
        val_true = convert_ck2dist(self.erg_ctrl_true.basis, ck_true, size=self.size)

        path_dr = np.stack(self.log['trajectory_dr'])[:self.tf, self.model_dr.explr_idx]
        ck_dr = convert_traj2ck(self.erg_ctrl_dr.basis, path_dr)
        val_dr = convert_ck2dist(self.erg_ctrl_dr.basis, ck_dr, size=self.size)

        plt.cla()
        plt.clf()
        plt.close()
        fig = plt.figure()

        ax1 = fig.add_subplot(131)
        ax1.contourf(*xy, val_true.reshape(50, 50), levels=20)
        ax1.set_aspect('equal', 'box')
        ax1.set_title('Actual Path Statistics')

        ax2 = fig.add_subplot(132)
        ax2.contourf(*xy, vals, levels=20)
        ax2.set_aspect('equal', 'box')
        ax2.set_title('Spatial Distribution')

        ax3 = fig.add_subplot(133)
        ax3.contourf(*xy, val_dr.reshape(50, 50), levels=20)
        ax3.set_aspect('equal', 'box')
        ax3.set_title('Dead Reckoning Path Statistics')

        if save is not None:
            plt.savefig(save)

        plt.show()
        plt.cla()
        plt.clf()
        plt.close()

        return fig
