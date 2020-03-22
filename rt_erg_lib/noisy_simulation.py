import matplotlib.pyplot as plt
from matplotlib import animation
import autograd.numpy as np
from numpy import sin, cos
from .utils import convert_ck2dist, convert_traj2ck
from tqdm import tqdm
import time

"""
belief trajectory is ergodic, but ground truth is not
"""

class noisy_simulation():
    def __init__(self, size, init_state, noise, t_dist, model, erg_ctrl, true_env, belief_env, tf):
        self.size = size
        self.init_state = init_state
        self.noise = noise
        self.erg_ctrl = erg_ctrl
        self.true_env = true_env
        self.belief_env = belief_env
        self.tf = tf
        self.t_dist = t_dist
        self.model = model
        self.exec_times = np.zeros(tf)

    def start(self, report=False):
        self.log = {'truth': [], 'belief': [], 'ctrl': [], 'nctrl':[]}
        belief_state = self.belief_env.reset(self.init_state)
        true_state = self.true_env.reset(self.init_state)
        for t in tqdm(range(self.tf)):
            start_time = time.time()
            ctrl = self.erg_ctrl(belief_state)
            belief_state = self.belief_env.step(ctrl)
            true_state = self.true_env.noisy_step(ctrl, self.noise)
            self.log['belief'].append(belief_state)
            self.log['truth'].append(true_state)
            self.log['ctrl'].append(ctrl)
            self.exec_times[t] = time.time() - start_time
        print("simulation finished.")
        if report:
            print(self.exec_times[1:10])
            print("mean execution time: {0:.6f}(s), standard deviation: {1:.6f}(s)".format(np.mean(self.exec_times),
                                                                                           np.std(self.exec_times)))

    def plot(self, point_size=1):
        [xy, vals] = self.t_dist.get_grid_spec()
        plt.contourf(*xy, vals, levels=20)
        true_xt = np.stack(self.log['truth'])
        belief_xt = np.stack(self.log['belief'])
        plt.scatter(belief_xt[:self.tf, 0], belief_xt[:self.tf, 1], s=point_size, c='red')
        plt.scatter(true_xt[:self.tf, 0], true_xt[:self.tf, 1], s=point_size, c='cyan')
        ax = plt.gca()
        ax.set_aspect('equal', 'box')

        plt.show()
        # plt.cla()
        # plt.clf()
        # plt.close()
        return plt.gcf()

    def animate(self, point_size=1, show_label=False, show_traj=True, rate=50):
        [xy, vals] = self.t_dist.get_grid_spec()
        true_xt = np.stack(self.log['truth'])
        belief_xt = np.stack(self.log['belief'])
        plt.contourf(*xy, vals, levels=20)
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        fig = plt.gcf()
        points1 = ax.scatter([], [], s=point_size, c='red')
        points2 = ax.scatter([], [], s=point_size, c='cyan')

        def sub_animate(i):
            if show_traj:
                points1.set_offsets(np.array([belief_xt[:i, 0], belief_xt[:i, 1]]).T)
                points2.set_offsets(np.array([true_xt[:i, 0], true_xt[:i, 1]]).T)
            else:
                points1.set_offsets(np.array([[belief_xt[i, 0]], [belief_xt[i, 1]]]).T)
                points2.set_offsets(np.array([[true_xt[i, 0]], [true_xt[i, 1]]]).T)
                # if len(true_xt[1, :])/2 == 3:
                #     plt.arrow(belief_xt[i, 0], belief_xt[i, 1], cos(belief_xt[i, 2]), sin(belief_xt[i, 2]))
            if show_label:
                bx = round(belief_xt[i, 0], 2)
                by = round(belief_xt[i, 1], 2)
                bth = round(belief_xt[i, 2], 2)
                tx = round(true_xt[i, 0], 2)
                ty = round(true_xt[i, 1], 2)
                tth = round(true_xt[i, 2], 2)
                quiver1 = ax.quiver(bx, by, cos(bth), sin(bth), color='red')
                quiver2 = ax.quiver(tx, ty, cos(tth), sin(tth), color='cyan')
                return [points1, points2, quiver1, quiver2]
            else:
                return [points1, points2]

        anim = animation.FuncAnimation(fig, sub_animate, frames=self.tf, interval=(1000 / rate), blit=True)

        plt.show()
        # plt.cla()
        # plt.clf()
        # plt.close()
        return anim

    def path_reconstruct(self):
        xy, vals = self.t_dist.get_grid_spec()
        belief_path = np.stack(self.log['belief'])[:self.tf, self.model.explr_idx]
        true_path = np.stack(self.log['truth'])[:self.tf, self.model.explr_idx]
        belief_ck = convert_traj2ck(self.erg_ctrl.basis, belief_path)
        true_ck = convert_traj2ck(self.erg_ctrl.basis, true_path)
        belief_val = convert_ck2dist(self.erg_ctrl.basis, belief_ck, size=self.size)
        true_val = convert_ck2dist(self.erg_ctrl.basis, true_ck, size=self.size)

        plt.cla()
        plt.clf()
        plt.close()
        fig = plt.figure()

        ax1 = fig.add_subplot(131)
        ax1.contourf(*xy, belief_val.reshape(50, 50), levels=20)
        ax1.set_aspect('equal', 'box')
        ax1.set_title('Belief Path Statistics')

        ax2 = fig.add_subplot(132)
        ax2.contourf(*xy, vals, levels=20)
        ax2.set_aspect('equal', 'box')
        ax2.set_title('Spatial Distribution')

        ax3 = fig.add_subplot(133)
        ax3.contourf(*xy, true_val.reshape(50, 50), levels=20)
        ax3.set_aspect('equal', 'box')
        ax3.set_title('True Path Statistics')

        plt.show()
        plt.cla()
        plt.clf()
        plt.close()
        return fig