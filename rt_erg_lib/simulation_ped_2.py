import matplotlib.pyplot as plt
from matplotlib import animation
import autograd.numpy as np
from .utils import *
from tqdm import tqdm
import time
from numpy import sin, cos, sqrt

class simulation():
    def __init__(self, size, init_state, model, erg_ctrl, env, tf, ped_data, space, goals=None):
        self.size = size
        self.init_state = init_state
        self.erg_ctrl = erg_ctrl
        self.env = env
        self.tf = tf
        self.model = model

        self.ped_data = ped_data
        self.space = space
        grid = np.meshgrid(*[np.linspace(0, size, 50) for _ in range(2)])
        self.grid = np.c_[grid[0].ravel(), grid[1].ravel()] # 2*N array

        self.goal_vals = 0
        self.goals = goals
        cov = np.array([1.2, 1.2])
        if goals is not None:
            for mean in goals:
                print("mean: ", mean)
                innerds = np.sum((self.grid-mean)**2 / cov, 1)
                self.goal_vals += np.exp(-innerds/2.0)
            self.goal_vals /= np.sum(self.goal_vals)
            self.goal_vals *= 10.02


    def start(self):
        self.log = {'trajectory': [], 'ped_state': [],
                    'dist_vals':[]}
        state = self.env.reset(self.init_state)

        print(self.ped_data.shape)

        for t in tqdm(range(self.tf)):
            # ped data process
            # ped_state = self.ped_data[t]
            if t == 0:
                ped_state = self.ped_data[t]
                # dist_vals = self.distance_field5(self.grid, ped_state, self.space) + self.goal_vals
                dist_vals = self.goal_vals
                self.erg_ctrl.phik = convert_phi2phik(self.erg_ctrl.basis,
                                                  dist_vals,
                                                  self.grid)
            # ergodic control
            ctrl = self.erg_ctrl(state)
            state = self.env.step(ctrl)
            self.log['trajectory'].append(state)
            self.log['ped_state'].append(ped_state)
            if t == 0:
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

    def distance_field5(self, grid, state, space, dist_threshold=1.0):
        start = time.time()
        grid_x = grid[:,0]
        grid_y = grid[:,1]
        space = np.array(space).reshape(-1,2)
        space_x = space[:,0]
        space_y = space[:,1]
        state_x = np.concatenate((state[:,0], space_x))[:, np.newaxis]
        state_y = np.concatenate((state[:,1], space_y))[:, np.newaxis]
        diff_x = grid_x - state_x
        diff_y = grid_y - state_y
        dist_xy = np.sqrt(diff_x**2 + diff_y**2)

        dist_flag = dist_xy > dist_threshold
        dist_flag = dist_flag.astype(int)
        dist_xy *= dist_flag

        dist_val = dist_xy.min(axis=0)
        dist_val /= np.sum(dist_val)
        return dist_val

    def plot(self, point_size=1, save=None):
        xy = []
        for g in self.grid.T:
            xy.append(
                np.reshape(g, newshape=(50, 50))
        )
        vals = self.log['dist_vals'][0].reshape(50, 50)

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
        xt = np.stack(self.log['trajectory'])

        # plt.contourf(*xy, vals, levels=20)
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-0.5, 15.5)
        ax.set_ylim(-0.5, 15.5)

        for boundary in self.space:
        	ax.scatter(boundary[:,0], boundary[:,1], s=5, c='k')

        if self.goals is not None:
        	ax.scatter(self.goals[:,0], self.goals[:,1], s=30, c='k', marker='+')

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
        xt = np.stack(self.log['trajectory'])
        fig = plt.figure()

        # plt.contourf(*xy, vals, levels=20)
        ax = fig.add_subplot(121)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-0.5, 15.5)
        ax.set_ylim(-0.5, 15.5)

        for boundary in self.space:
        	ax.scatter(boundary[:,0], boundary[:,1], s=5, c='k')

        if self.goals is not None:
        	ax.scatter(self.goals[:,0], self.goals[:,1], s=30, c='k', marker='+')

        points = ax.scatter([], [], s=point_size, c='r')
        peds = ax.scatter([], [], s=point_size, c='b')

        ax2 = fig.add_subplot(122)
        ax2.set_aspect('equal', 'box')
        points2 = ax2.scatter([], [], s=point_size, c='r')
        xy = []
        for g in self.grid.T:
            xy.append(
                np.reshape(g, newshape=(50, 50))
        )

        print("dist_vals shape: ", len(self.log['dist_vals']))

        def sub_animate(i):
            snapshot = self.ped_data[i]
            peds.set_offsets(snapshot[:,0:2])
            if(show_traj):
                points.set_offsets(np.array([xt[:i, 0], xt[:i, 1]]).T)
            else:
                points.set_offsets(np.array([[xt[i, 0]], [xt[i, 1]]]).T)

            ax2.clear()
            # ax2.set_aspect('equal', 'box')
            ax2.plot(xt[i, 0], xt[i, 1], 'ro')
            ax2.contourf(*xy, self.log['dist_vals'][0].reshape(50,50), levels=25)

            ret = [points, peds]
            return ret

        anim = animation.FuncAnimation(fig, sub_animate, frames=self.tf, interval=(1000/rate))#, blit=True)
        if save is not None:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=40, metadata=dict(artist='simulation_slam'), bitrate=5000)
            anim.save(save, writer=writer)
        plt.show()
        # return anim


    def path_reconstruct(self, save=None):
        xy = []
        for g in self.grid.T:
            xy.append(
                np.reshape(g, newshape=(50, 50))
        )

        vals = self.log['dist_vals'][0].reshape(50, 50)
        path = np.stack(self.log['trajectory'])[:self.tf, self.model.explr_idx]
        ck = convert_traj2ck(self.erg_ctrl.basis, path)
        val = convert_ck2dist(self.erg_ctrl.basis, ck, size=self.size)
        plt.contourf(*xy, val.reshape(50,50), levels=20)
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        if save is not None:
            plt.savefig(save)
        plt.show()
