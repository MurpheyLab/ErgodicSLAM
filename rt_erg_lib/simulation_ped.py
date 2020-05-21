import matplotlib.pyplot as plt
from matplotlib import animation
import autograd.numpy as np
from .utils import *
from tqdm import tqdm
import time
from numpy import sin, cos, sqrt

class simulation():
    def __init__(self, size, init_state, t_dist, model, erg_ctrl, env, tf, ped_data):
        self.size = size
        self.init_state = init_state
        self.erg_ctrl = erg_ctrl
        self.env = env
        self.tf = tf
        self.t_dist = t_dist
        self.model = model

        self.ped_data = ped_data
        grid = np.meshgrid(*[np.linspace(0, size, 50) for _ in range(2)])
        self.grid = np.c_[grid[0].ravel(), grid[1].ravel()] # 2*N array

    def start(self):
        self.log = {'trajectory': [], 'ped_state': [],
                    'dist_vals':[]}
        state = self.env.reset(self.init_state)

        print(self.ped_data.shape)

        for t in tqdm(range(self.tf)):
            # ped data process
            ped_state = self.ped_data[t]
            dist_vals = self.distance_field(self.grid, ped_state)
            self.erg_ctrl.phik = convert_phi2phik(self.erg_ctrl.basis,
                                                  dist_vals,
                                                  self.grid)
            # ergodic control
            ctrl = self.erg_ctrl(state)
            state = self.env.step(ctrl)
            self.log['trajectory'].append(state)
            self.log['ped_state'].append(ped_state)
            self.log['dist_vals'].append(dist_vals)
        print("simulation finished.")

    def distance_field(self, grid, state):
        grid_x = grid[:,0]
        grid_y = grid[:,1]
        state_x = state[:,0][:, np.newaxis]
        state_y = state[:,1][:, np.newaxis]
        diff_x = grid_x - state_x
        diff_y = grid_y - state_y
        diff_xy = np.sqrt(diff_x**2 + diff_y**2)
        dist_xy = diff_xy.min(axis=0)
        return dist_xy

    def plot(self, point_size=1, save=None):
        [xy, vals] = self.t_dist.get_grid_spec()
        plt.contourf(*xy, vals, levels=20)
        xt = np.stack(self.log['trajectory'])
        plt.scatter(xt[:self.tf, 0], xt[:self.tf, 1], s=point_size, c='red')
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        if save is not None:
            plt.savefig(save)
        plt.show()
        # return plt.gcf()

    def animate(self, point_size=1, show_label=False, show_traj=True, save=None, rate=50):
        [xy, vals] = self.t_dist.get_grid_spec()
        xt = np.stack(self.log['trajectory'])

        # plt.contourf(*xy, vals, levels=20)
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-0.5, 15.5)
        ax.set_ylim(-0.5, 15.5)

        fig = plt.gcf()
        points = ax.scatter([], [], s=point_size, c='r')
        peds = ax.scatter([], [], s=point_size, c='b')
        def sub_animate(i):
            snapshot = self.ped_data[i]
            peds.set_offsets(snapshot[:,0:2])
            if(show_traj):
                points.set_offsets(np.array([xt[:i, 0], xt[:i, 1]]).T)
            else:
                points.set_offsets(np.array([[xt[i, 0]], [xt[i, 1]]]).T)

            if show_label:
                cx = round(xt[i, 0], 2)
                cy = round(xt[i, 1], 2)
                cth = round(xt[i, 2], 2)
                quiver1 = ax.quiver(cx, cy, cos(cth), sin(cth), color='red')
                if i == self.tf-1:
                    quiver1.remove()
                    points.set_offsets(np.array([[xt[i, 0]], [xt[i, 1]]]).T)
                    ret = [points, peds]
                else:
                    ret = [points, peds, quiver1]
                return ret
            else:
                ret = [points, peds]
                return ret

        anim = animation.FuncAnimation(fig, sub_animate, frames=self.tf, interval=(1000/rate), blit=True)
        if save is not None:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=40, metadata=dict(artist='simulation_slam'), bitrate=5000)
            anim.save(save, writer=writer)
        plt.show()
        # return anim

    def animate2(self, point_size=1, show_label=False, show_traj=True, save=None, rate=50):
        [xy, vals] = self.t_dist.get_grid_spec()
        xt = np.stack(self.log['trajectory'])

        fig = plt.figure()
        # plt.contourf(*xy, vals, levels=20)
        ax1 = fig.add_subplot(111)
        ax1.set_aspect('equal', 'box')
        ax1.set_xlim(-0.5, 15.5)
        ax1.set_ylim(-0.5, 15.5)

        xy = []
        for g in self.grid.T:
            xy.append(np.reshape(g, newshape=(50,50)))

        def sub_animate(i):
            ax1.clear()
            # ax1.cla()
            snapshot = self.ped_data[i]
            peds = ax1.scatter(snapshot[:,0], snapshot[:,1],
                               s=point_size, c='w')
            if(show_traj):
                points1 = ax1.scatter(xt[:i, 0], xt[:i, 1],
                                      s=point_size, c='r')
            else:
                points1 = ax1.scatter([xt[i,0]], [xt[i,1]],
                                      s=point_size, c='r')
            ax1.contourf(*xy, self.log['dist_vals'][i].reshape(50,50),
                         levels=25)
            ret = [points1, peds]
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
        path = np.stack(self.log['trajectory'])[:self.tf, self.model.explr_idx]
        ck = convert_traj2ck(self.erg_ctrl.basis, path)
        val = convert_ck2dist(self.erg_ctrl.basis, ck, size=self.size)
        plt.contourf(*xy, val.reshape(50,50), levels=20)
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        if save is not None:
            plt.savefig(save)
        plt.show()
