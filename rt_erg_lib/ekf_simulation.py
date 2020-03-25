import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
import autograd.numpy as np
from .utils import convert_ck2dist, convert_traj2ck, normalize_angle
from tqdm import tqdm
import time
import numpy as np
from numpy import sin, cos, sqrt
import math
from math import pi

class simulation_slam():
    def __init__(self, size, init_state, t_dist, model_true, erg_ctrl_true, env_true, model_dr, erg_ctrl_dr, env_dr, tf, landmarks, sensor_range, motion_noise, measure_noise):
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

        self.motion_noise = motion_noise
        self.R = np.diag(motion_noise) ** 2
        # self.R = np.sqrt(np.diag(motion_noise))
        print("R mat: ", self.R)
        self.measure_noise = measure_noise
        self.Q = np.diag(measure_noise) ** 2
        # self.Q = np.sqrt(np.diag(measure_noise))
        print("Q mat: ", self.Q)
        self.observed_landmarks = np.zeros(self.landmarks.shape[0])

    def start(self, report=False):
        #########################
        # initialize mean and covariance matrix
        #########################
        self.nLandmark = self.landmarks.shape[0]     # number of landmarks
        self.nStates = self.init_state.shape[0]
        self.dim = self.nStates + 2 * self.nLandmark
        mean = np.zeros(self.dim)
        cov = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            if i < self.init_state.shape[0]:
                mean[i] = self.init_state[i]
                cov[i, i] = 0
            else:
                cov[i, i] = 99999999

        ##########################
        # simulation loop
        ##########################
        self.log = {'trajectory_true': [], 'trajectory_dr': [], 'true_landmarks': [], 'observations': [], 'mean': [], 'covariance': []}
        state_true = self.env_true.reset(self.init_state)
        state_dr = self.env_dr.reset(self.init_state)

        for t in tqdm(range(self.tf)):
            #########################
            # generate control and measurement data
            #########################
            # this is what robot thinks it's doing
            ctrl = self.erg_ctrl_dr(state_dr)
            state_dr = self.env_dr.step(ctrl)
            self.log['trajectory_dr'].append(state_dr)

            # this is what robot is actually doing (if we don't correct it)
            state_true = self.env_true.noisy_step(ctrl, self.motion_noise)
            self.log['trajectory_true'].append(state_true)

            # observation model
            true_landmarks = []
            observations = []
            for i in range(self.nLandmark):
                item = self.landmarks[i]
                dist = sqrt((item[0]-state_true[0])**2 + (item[1]-state_true[1])**2)
                if(dist <= self.sensor_range):
                    true_landmarks.append(i)
                    noisy_observation = self.range_bearing(i, state_true, item)
                    noisy_observation[1:] += self.measure_noise * np.random.randn(2)
                    observations.append(noisy_observation)
            self.log['true_landmarks'].append(true_landmarks)
            self.log['observations'].append(np.array(observations))
            print(self.log['observations'][t])

            #########################
            # EKF SLAM
            #########################
            mean[2] = normalize_angle(mean[2])
            predict_mean, predict_cov = self.ekf_slam_prediction(mean, cov, ctrl, self.R)
            predict_mean[2] = normalize_angle(predict_mean[2])
            predict_mean, predict_cov = self.ekf_correction(predict_mean, predict_cov, observations, self.Q)
            predict_mean[2] = normalize_angle(predict_mean[2])

            self.log['mean'].append(predict_mean)
            self.log['covariance'].append(predict_cov)

            mean = predict_mean
            cov = predict_cov

            print("\nest:\t", mean[0:2])
            print("true:\t", state_true[0:2])

        print("simulation finished.")

    def range_bearing(self, id, agent, landmark):
        delta = landmark - agent[0:self.nStates-1]
        range = np.sqrt(np.dot(delta.T, delta))
        bearing = math.atan2(delta[1], delta[0]) - agent[2]
        bearing = normalize_angle(bearing)
        return np.array([id, range, bearing])

    # def ekf_slam_prediction(self, mean, cov, ctrl, R):
    #     # update mean
    #     predict_mean = mean.copy()
    #     predict_mean[0:self.nStates] += self.env_true.f(mean[0:self.nStates], ctrl) * self.env_true.dt
    #
    #     # update covariance matrix
    #     Gx = self.env_true.Gx(mean[0:self.nStates], ctrl)
    #     G = np.block([
    #         [Gx, np.zeros((self.nStates, 2*self.nLandmark))],
    #         [np.zeros((2*self.nLandmark, self.nStates)), np.eye(2*self.nLandmark)]
    #     ])
    #     R = np.block([
    #         [self.R, np.zeros((self.nStates, 2*self.nLandmark))],
    #         [np.zeros((2*self.nLandmark, self.nStates)), np.eye(2*self.nLandmark)]
    #     ])
    #     predict_cov = np.dot(np.dot(G, cov), G.T) + R
    #
    #     return predict_mean, predict_cov

    def ekf_slam_prediction(self, mean, cov, ctrl, R):
        # generate matrix F
        F = np.block([np.eye(self.nStates), np.zeros((self.nStates, 2 * self.nLandmark))])

        # update predicted mean
        predict_mean = mean.copy()
        predict_mean[0:self.nStates] += self.env_true.f(mean[0:self.nStates], ctrl) * self.env_true.dt

        # calculate matrix G for updating predicted covariance matrix
        Jacobian = np.array([[0, 0, -sin(mean[2]) * ctrl[0] * self.env_true.dt],
                             [0, 0,  cos(mean[2]) * ctrl[0] * self.env_true.dt],
                             [0, 0,  0]])
        G = np.eye(cov.shape[0]) + np.dot(np.dot(F.T, Jacobian), F)

        # update predicted covariance
        predict_cov = np.dot(np.dot(G, cov), G.T) + np.dot(np.dot(F.T, R), F)

        # return
        return predict_mean, predict_cov

    def ekf_correction(self, predict_mean, predict_cov, z, Q):
        # initialize mean and cov
        mean = predict_mean.copy()
        cov = predict_cov.copy()

        # iterate each observed landmark
        for obs in z:
            # extract measurement data
            id = int(obs[0])
            measurement = obs[1:]
            # if landmark not observed, initialize mean
            if(self.observed_landmarks[id] == 0):
                self.observed_landmarks[id] = 1
                loc_x = mean[0] + measurement[0] * cos(mean[2] + measurement[1])
                loc_y = mean[1] + measurement[0] * sin(mean[2] + measurement[1])
                mean[2 + 2 * id + 1] = loc_x
                mean[2 + 2 * id + 2] = loc_y
            est_landmark = np.array([mean[2 + 2 * id + 1], mean[2 + 2 * id + 2]])
            # get expected measurement (range-bearing)
            delta = est_landmark - mean[0:2]
            zi = self.range_bearing(id, mean[0:3], est_landmark)
            zi = zi[1:]
            q = zi[0]**2
            q_sqrt = zi[0]
            # generate matrix F
            F = np.zeros((5, 3 + 2 * self.nLandmark))
            F[0, 0] = 1
            F[1, 1] = 1
            F[2, 2] = 1
            F[3, 2 + 2 * id + 1] = 1
            F[4, 2 + 2 * id + 2] = 1
            # calculate Jacobian matrix H of the measurement model
            temp = np.array([
                [-q_sqrt * delta[0], -q_sqrt * delta[1], 0, q_sqrt * delta[0], q_sqrt * delta[1]],
                [delta[1], -delta[0], -q, -delta[1], delta[0]]
            ])
            H = (1/zi[0]**2) * np.dot(temp, F)
            # calculate Kalman gain: matrix K
            mat1 = np.dot(cov, H.T)
            mat2 = np.dot(np.dot(H, cov), H.T)
            mat3 = np.linalg.inv(mat2 + Q)
            K = np.dot(mat1, mat3)
            # update mean and covariance matrix
            if measurement[1] * zi[1] < 0:
                print("\nmeasurement:\t", measurement)
                print("zi: \t\t", zi)
            diff_z = measurement - zi
            diff_z[1] = normalize_angle(diff_z[1])
            mean += np.dot(K, diff_z)
            cov -= np.dot(np.dot(K, H), cov)

        return mean, cov

    def generate_ellipse(self, x, y, theta, a, b):
        NPOINTS = 100
        # compose point vector
        ivec = np.arange(0, 2*pi, 2*pi/NPOINTS)
        p = np.zeros((2, NPOINTS))
        p[0, :] = a * cos(ivec)
        p[1, :] = b * sin(ivec)

        # translate and rotate
        R = np.array([
            [cos(theta), -sin(theta)],
            [sin(theta),  cos(theta)]
        ])
        p = np.dot(R, p)
        p[0, :] += x
        p[1, :] += y

        return p

    def generate_cov_ellipse(self, mean, cov, alpha=1):
        sxx = cov[0,0]
        syy = cov[1,1]
        sxy = cov[0,1]
        a = alpha * np.sqrt(0.5 * (sxx + syy + np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2)))
        b = alpha * np.sqrt(0.5 * (sxx + syy - np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2)))
        theta = mean[2] # % (2*pi)
        # print(a, b, theta)

        p = self.generate_ellipse(mean[0], mean[1], theta, a, b)
        return p

    def plot(self, point_size=1, save=None):
        [xy, vals] = self.t_dist.get_grid_spec()
        plt.contourf(*xy, vals, levels=20)

        xt_true = np.stack(self.log['trajectory_true'])
        traj_true = plt.scatter(xt_true[:self.tf, 0], xt_true[:self.tf, 1], s=point_size, c='red')
        xt_dr = np.stack(self.log['trajectory_dr'])
        traj_dr = plt.scatter(xt_dr[:self.tf, 0], xt_dr[:self.tf, 1], s=point_size, c='cyan')
        xt_est = np.stack(self.log['mean'])
        traj_est = plt.scatter(xt_est[:self.tf, 0], xt_est[:self.tf, 1], s=point_size, c='green')

        plt.legend([traj_true, traj_dr, traj_est], ['True Path', 'Dead Reckoning Path', 'Estimated Path'])

        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        if save is not None:
            plt.savefig(save)
        plt.show()
        # return plt.gcf()

    def animate(self, point_size=1, show_traj=True, save=None, rate=50):
        [xy, vals] = self.t_dist.get_grid_spec()
        plt.contourf(*xy, vals, levels=20)
        plt.scatter(self.landmarks[:, 0], self.landmarks[:, 1], color='white')
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        fig = plt.gcf()

        xt_true = np.stack(self.log['trajectory_true'])
        points_true = ax.scatter([], [], s=point_size, c='red')
        xt_dr = np.stack(self.log['trajectory_dr'])
        points_dr = ax.scatter([], [], s=point_size, c='cyan')
        mean_est = np.stack(self.log['mean'])
        xt_est = mean_est[:, 0:3]
        points_est = ax.scatter([], [], s=point_size, c='green')

        observation_lines = []
        for id in range(self.landmarks.shape[0]):
            observation_lines.append(ax.plot([], [], color='orange'))

        agent_ellipse = ax.scatter([], [], s=point_size, c='white')

        plt.legend([points_true, points_dr, points_est], ['True Path', 'Dead Reckoning Path', 'Estimated Path'])

        sensor_points = []
        for id in range(self.landmarks.shape[0]):
            sensor_point = ax.plot([], [], color='orange')
            sensor_points.append(sensor_point)

        def sub_animate(i):
            # visualize agent location / trajectory
            if(show_traj):
                points_true.set_offsets(np.array([xt_true[:i, 0], xt_true[:i, 1]]).T)
                points_dr.set_offsets(np.array([xt_dr[:i, 0], xt_dr[:i, 1]]).T)
                points_est.set_offsets(np.array([xt_est[:i, 0], xt_est[:i, 1]]).T)
            else:
                points_true.set_offsets(np.array([[xt_true[i, 0]], [xt_true[i, 1]]]).T)
                points_true.set_offsets(np.array([[xt_dr[i, 0]], [xt_dr[i, 1]]]).T)
                points_est.set_offsets(np.array([[xt_est[i, 0]], [xt_est[i, 1]]]).T)

            # visualize agent covariance matrix as ellipse
            mean = self.log['mean'][i]
            mean = mean[0:self.nStates]
            cov = self.log['covariance'][i]
            cov = cov[0:self.nStates-1, 0:self.nStates-1]
            # print(cov)
            p_agent = self.generate_cov_ellipse(mean, cov, alpha=1)
            agent_ellipse.set_offsets(np.array([p_agent[0, :], p_agent[1, :]]).T)

            # clear observation model visualization
            for point in sensor_points:
                point[0].set_xdata([])
                point[0].set_ydata([])

            # observation model visualization
            for observation in self.log['observations'][i]:
                id = int(observation[0])
                measurement = observation[1:]
                loc_x = xt_true[i, 0] + measurement[0] * cos(xt_true[i, 2] + measurement[1])
                loc_y = xt_true[i, 1] + measurement[0] * sin(xt_true[i, 2] + measurement[1])
                sensor_points[id][0].set_xdata([xt_true[i, 0], loc_x])
                sensor_points[id][0].set_ydata([xt_true[i, 1], loc_y])

            # return matplotlib objects for animation
            ret = [points_true, points_dr, agent_ellipse, points_est]
            for item in sensor_points:
                ret.append(item[0])
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

        path_est = np.stack(self.log['mean'])[:self.tf, self.model_dr.explr_idx]
        ck_est = convert_traj2ck(self.erg_ctrl_dr.basis, path_est)
        val_est = convert_ck2dist(self.erg_ctrl_dr.basis, ck_est, size=self.size)

        plt.cla()
        plt.clf()
        plt.close()
        fig = plt.figure()

        ax1 = fig.add_subplot(221)
        ax1.contourf(*xy, vals, levels=20)
        ax1.set_aspect('equal', 'box')
        ax1.set_title('Spatial Distribution')

        ax2 = fig.add_subplot(222)
        ax2.contourf(*xy, val_true.reshape(50, 50), levels=20)
        ax2.set_aspect('equal', 'box')
        ax2.set_title('Actual Path Statistics')

        ax3 = fig.add_subplot(223)
        ax3.contourf(*xy, val_dr.reshape(50, 50), levels=20)
        ax3.set_aspect('equal', 'box')
        ax3.set_title('Dead Reckoning Path Statistics')

        ax4 = fig.add_subplot(224)
        ax4.contourf(*xy, val_est.reshape(50, 50), levels=20)
        ax4.set_aspect('equal', 'box')
        ax4.set_title('Estimated Path Statistics')

        if save is not None:
            plt.savefig(save)

        plt.show()
        plt.cla()
        plt.clf()
        plt.close()

        return fig
