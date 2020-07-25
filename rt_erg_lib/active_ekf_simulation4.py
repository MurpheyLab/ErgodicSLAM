'''
ekf implemenatation with varying dimension
mutual information
plot through grids instead of contour
'''

import matplotlib.pyplot as plt
from matplotlib import animation
import autograd.numpy as np
from .target_dist import TargetDist
from .utils import *
from tqdm import tqdm
from numpy import sin, cos, sqrt
import math
from math import pi
from tempfile import TemporaryFile
import copy
from time import time
from scipy.linalg import block_diag


class simulation_slam():
    def __init__(self, size, init_state, t_dist, model_true, erg_ctrl_true, env_true, model_dr, erg_ctrl_dr, env_dr, tf,
                 landmarks, sensor_range, motion_noise, measure_noise, switch):
        self.size = size
        self.init_state = init_state
        self.tf = tf
        self.init_t_dist = copy.copy(t_dist)
        self.t_dist = t_dist
        self.exec_times = np.zeros(tf)
        self.landmarks = landmarks
        self.sensor_range = sensor_range
        self.num_pts = 50

        self.erg_ctrl_true = erg_ctrl_true
        self.env_true = env_true
        self.model_true = model_true

        self.erg_ctrl_dr = erg_ctrl_dr
        self.env_dr = env_dr
        self.model_dr = model_dr

        self.motion_noise = motion_noise
        self.R = np.diag(motion_noise) ** 2
        self.measure_noise = measure_noise
        self.Q = np.diag(measure_noise) ** 2
        self.mcov_inv = np.linalg.inv(self.Q)
        self.observed_landmarks = np.zeros(self.landmarks.shape[0])
        self.new_observed = None
        self.threshold = 99999999

        self.og_vals = np.ones((self.num_pts, self.num_pts)) * 0.5
        self.raw_grid = np.meshgrid(*[np.linspace(0, self.size, self.num_pts+1) for _ in range(2)])
        self.grid = [self.raw_grid[0][0:self.num_pts,0:self.num_pts], self.raw_grid[1][0:self.num_pts,0:self.num_pts]]
        self.init_phik = convert_phi2phik(self.erg_ctrl_dr.basis, self.t_dist.target_grid_vals, self.t_dist.grid)

        self.lm_id = []
        self.curr_obsv = []
        self.switch = switch
        self.horizon = self.erg_ctrl_true.horizon

    def start(self, report=False, debug=False, update=1, update_threshold=1e-3, snapshot=0):
        #########################
        # initialize mean and covariance matrix
        #########################
        self.nLandmark = self.landmarks.shape[0]  # number of landmarks
        self.nStates = self.init_state.shape[0]
        self.dim = self.nStates + 2 * self.nLandmark

        mean = self.init_state.copy()
        cov = np.zeros((self.nStates, self.nStates))

        ##########################
        # simulation loop
        ##########################
        self.log = {'tf': self.tf, 'trajectory_true': [], 'trajectory_dr': [], 'true_landmarks': [], 'observations': [], 'mean': [],
                'covariance': [], 'planning_mean': [], 'planning_cov': [], 'target_dist': [], 'error':[], 'uncertainty':[], 'metric_true':[], 'metric_est':[], 'metric_error':[], 'landmarks':self.landmarks, 'trajectory_slam':[], 'erg_ctrls':[]}
        state_true = self.env_true.reset(self.init_state)
        state_dr = self.env_dr.reset(self.init_state)

        print('start simulation ... update mechanism: ', update)
        self.curr_t = 0
        for t in tqdm(range(self.tf)):
            self.curr_t = t
            #########################
            # generate control and measurement data
            #########################
            # this is what robot thinks it's doing
            if t<self.switch:  # debug mode: robot runs a circle
                ctrl = np.array([2.5, 0.4])
            else:
                ctrl, erg_ctrls = self.erg_ctrl_dr(mean[0:self.nStates], get_useq=True)
                self.log['erg_ctrls'].append(erg_ctrls)
            state_dr = self.env_dr.step(ctrl)
            self.log['trajectory_dr'].append(state_dr)

            # this is what robot is actually doing (if we don't correct it)
            state_true = self.env_true.noisy_step(ctrl, self.motion_noise)
            self.log['trajectory_true'].append(state_true)

            # observation model
            true_landmarks = []
            observations = []
            self.curr_obsv = []
            for i in range(self.nLandmark):
                item = self.landmarks[i]
                dist = sqrt((item[0] - state_true[0]) ** 2 + (item[1] - state_true[1]) ** 2)
                if (dist <= self.sensor_range):
                    true_landmarks.append(i)
                    noisy_observation = self.range_bearing(state_true, item)
                    noisy_observation += self.measure_noise * np.random.randn(2)
                    observations.append(noisy_observation)
                    self.curr_obsv.append(i)

                    if i in self.lm_id:
                        pass
                    else:
                        self.lm_id.append(i)
                        lm = self.observe_landmark(mean[0:3], noisy_observation)
                        mean = np.concatenate((mean, lm))
                        cov = np.block([
                                [cov, np.zeros((cov.shape[0],2))],
                                [np.zeros((2,cov.shape[0])), np.eye(2)*99999999.]
                            ])
            observations = np.array(observations)

            self.log['true_landmarks'].append(true_landmarks)
            self.log['observations'].append(np.array(observations))

            ########################
            # Planning prediction
            ########################
            '''
            planning_predict_mean, planning_predict_cov = self.planning_prediction(mean, cov, ctrl, self.R, self.Q)
            self.log['planning_mean'].append(planning_predict_mean)
            self.log['planning_cov'].append(planning_predict_cov)
            '''

            #########################
            # EKF SLAM
            #   now we have estimation of states for (t-1),
            #   ctrl, and observations at (t)
            #########################
            G = np.eye(mean.shape[0])
            G[0][2] = -sin(mean[2]) * ctrl[0] * 0.1
            G[1][2] =  cos(mean[2]) * ctrl[0] * 0.1
            num_lm = len(self.lm_id)
            BigR = np.block([
                    [self.R, np.zeros((3, 2*num_lm))],
                    [np.zeros((2*num_lm, 3)), np.zeros((2*num_lm, 2*num_lm))]
                ])
            cov = G @ cov @ G.T + BigR

            # process model for ekf prediction
            g = np.zeros(mean.shape[0])
            g[0] = cos(mean[2]) * ctrl[0]
            g[1] = sin(mean[2]) * ctrl[0]
            g[2] = ctrl[1]
            mean += g * 0.1
            mean[2] = normalize_angle(mean[2])

            # use observations for ekf correction
            num_obsv = len(self.curr_obsv)
            H = np.zeros((2*num_obsv, mean.shape[0]))
            r = mean[0:3].copy()
            lm_id = np.array(self.lm_id)
            ref_observations = []
            for i in range(num_obsv):
                idx = i*2
                lid = np.where(lm_id==self.curr_obsv[i])[0][0]
                lm = mean[3+lid*2 : 5+lid*2]
                zr = np.sqrt((r[0]-lm[0])**2 + (r[1]-lm[1])**2)

                H[idx][0]       = (r[0]-lm[0]) / zr
                H[idx][1]       = (r[1]-lm[1]) / zr
                H[idx][2]       = 0
                H[idx][3+2*lid] = -(r[0]-lm[0]) / zr
                H[idx][4+2*lid] = -(r[1]-lm[1]) / zr

                H[idx+1][0]         = -(r[1]-lm[1]) / zr**2
                H[idx+1][1]         =  (r[0]-lm[0]) / zr**2
                H[idx+1][2]         = -1
                H[idx+1][3+2*lid]   =  (r[1]-lm[1]) / zr**2
                H[idx+1][4+2*lid]   = -(r[0]-lm[0]) / zr**2

                ref_observations.append(self.range_bearing(r, lm))

            ref_observations = np.array(ref_observations)
            BigQ = block_diag(*[self.Q for _ in range(num_obsv)])

            mat1 = np.dot(cov, H.T)
            mat2 = np.dot(np.dot(H, cov), H.T)
            mat3 = np.linalg.inv(mat2 + BigQ)
            K = np.dot(mat1, mat3)

            # K = np.dot(np.dot(cov, H.T), np.linalg.inv(H @ cov @ H.T + BigQ))
            delta_z = observations - ref_observations
            if len(delta_z) == 0:
                pass
            else:
                delta_z[:,1] = normalize_angle(delta_z[:,1])
                delta_z = delta_z.reshape(-1)
                mean += K @ delta_z
                cov = cov - K @ H @ cov

            self.log['mean'].append(mean.copy())
            self.log['trajectory_slam'].append(mean[0:3].copy())
            self.log['covariance'].append(cov.copy())

            for i in range(self.num_pts):
                for j in range(self.num_pts):
                    cell = np.array([self.grid[0][i][j], self.grid[1][i][j]])
                    if (cell[0]-mean[0])**2 + (cell[1]-mean[1])**2 < 16 and self.og_vals[i][j] == 0.5:
                        self.og_vals[i][j] = 1.

            #########################
            # Record error and uncertainty for evaluation
            #########################
            self.log['uncertainty'].append(np.linalg.det(cov[0: self.nStates, 0: self.nStates]))
            self.log['error'].append(np.sqrt( (state_true[0]-mean[0])**2 + (state_true[1]-mean[1])**2 ))

            #########################
            # update target distribution and ergodic controller
            #########################
            # update target distribution with different update schemes
            if update == 0:
                self.erg_ctrl_dr.target_dist.update_intuitive(self.nStates, self.nLandmark, self.observed_landmarks, mean, cov)
            if update == 1:
                self.erg_ctrl_dr.target_dist.update_fim(self.nStates, self.nLandmark, self.observed_landmarks, mean, cov, self.mcov_inv, threshold=update_threshold)
            if update == 2:
                self.erg_ctrl_dr.target_dist.update_df_2(self.nStates, self.nLandmark, self.observed_landmarks, mean, cov, threshold=update_threshold)
            if update == 3:
                self.erg_ctrl_dr.target_dist.update_df_3(mean, cov, threshold=update_threshold)
            if update == 4:
                if np.sum(self.observed_landmarks) < self.observed_landmarks.shape[0]:
                    self.erg_ctrl_dr.target_dist.update_df_4(self.nStates, self.nLandmark, self.observed_landmarks, mean, cov, threshold=update_threshold, new_observed=self.new_observed)
                else:
                    self.erg_ctrl_dr.target_dist.update_df_3(self.nStates, self.nLandmark, self.observed_landmarks, mean, cov, threshold=update_threshold)
            if update == 5:
                self.erg_ctrl_dr.target_dist.update_fim_2(self.nStates, self.nLandmark, self.observed_landmarks, mean, cov, self.mcov_inv, threshold=update_threshold)
            if update == 6:
                self.erg_ctrl_dr.target_dist.update_mi_1(mean, cov, self.mcov_inv, self.og_vals)

            # update phi for ergodic controller
            self.erg_ctrl_dr.phik = convert_phi2phik(self.erg_ctrl_dr.basis, self.erg_ctrl_dr.target_dist.grid_vals, self.erg_ctrl_dr.target_dist.grid)
            # record target distribution for replay and visualization
            t_dist = copy.copy(self.erg_ctrl_dr.target_dist)
            self.log['target_dist'].append(t_dist)

        #######################
        # calculate and store ergodic metric during simulation
        #######################
        '''
        print('processing ergodic metric ...')
        xt_true = np.stack(self.log['trajectory_true'])
        mean_est = np.stack(self.log['mean'])
        xt_est = mean_est[:, 0:3]

        for i in tqdm(range(self.tf)):
            path_true = xt_true[:i+1, self.model_true.explr_idx]
            ck_true = convert_traj2ck(self.erg_ctrl_true.basis, path_true)
            erg_metric = self.erg_ctrl_dr.lamk * (ck_true - self.init_phik) ** 2
            erg_metric = np.sum( erg_metric.reshape(1,-1) )
            self.log['metric_true'].append(erg_metric)

            path_est = xt_est[:i+1, self.model_true.explr_idx]
            ck_est = convert_traj2ck(self.erg_ctrl_true.basis, path_est)
            erg_metric = self.erg_ctrl_dr.lamk * (ck_est - self.init_phik) ** 2
            erg_metric = np.sum( erg_metric.reshape(1,-1) )
            self.log['metric_est'].append(erg_metric)

        self.log['metric_true'] = np.array(self.log['metric_true'])
        self.log['metric_est'] = np.array(self.log['metric_est'])
        self.log['metric_error'] = np.abs( self.log['metric_true'] - self.log['metric_est'] )
        '''

        '''
        mean_est = np.stack(self.log['mean'])
        xt_est = mean_est[:, 0:3]
        for i in tqdm(range(self.tf)):
            path_est = xt_est[:i+1, self.model_true.explr_idx]
            ck_est = convert_traj2ck(self.erg_ctrl_true.basis, path_est)
            erg_metric = self.erg_ctrl_dr.lamk * (ck_est - self.erg_ctrl_dr.init_phik) ** 2
            erg_metric = np.sum( erg_metric.reshape(1,-1) )
            self.log['metric_est'].append(erg_metric)
        self.log['metric_est'] = np.array(self.log['metric_est'])

        self.log['metric_error'] = self.log['metric_true'] - self.log['metric_est']
        '''

        # return log for further visualization and evaluation
        print("simulation finished.")
        return self.log

    def range_bearing(self, agent, landmark):
        delta = landmark - agent[0:self.nStates - 1]
        rangee = np.sqrt(np.dot(delta.T, delta))
        bearing = math.atan2(delta[1], delta[0]) - agent[2]
        bearing = normalize_angle(bearing)
        return np.array([rangee, bearing])

    def observe_landmark(self, agent, obsv):
        lm_x = agent[0] + obsv[0] * cos(agent[2] + obsv[1])
        lm_y = agent[1] + obsv[0] * sin(agent[2] + obsv[1])
        return np.array([lm_x, lm_y])

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
                             [0, 0, cos(mean[2]) * ctrl[0] * self.env_true.dt],
                             [0, 0, 0]])
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
            if (self.observed_landmarks[id] == 0):
                self.observed_landmarks[id] = 1
                self.new_observed = id
                loc_x = mean[0] + measurement[0] * cos(mean[2] + measurement[1])
                loc_y = mean[1] + measurement[0] * sin(mean[2] + measurement[1])
                mean[2 + 2 * id + 1] = loc_x
                mean[2 + 2 * id + 2] = loc_y
            est_landmark = np.array([mean[2 + 2 * id + 1], mean[2 + 2 * id + 2]])
            # get expected measurement (range-bearing)
            delta = est_landmark - mean[0:2]
            zi = self.range_bearing(id, mean[0:3], est_landmark)
            zi = zi[1:]
            q = zi[0] ** 2
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
            H = (1 / zi[0] ** 2) * np.dot(temp, F)
            # calculate Kalman gain: matrix K
            mat1 = np.dot(cov, H.T)
            mat2 = np.dot(np.dot(H, cov), H.T)
            mat3 = np.linalg.inv(mat2 + Q)
            K = np.dot(mat1, mat3)
            # update mean and covariance matrix
            diff_z = measurement - zi
            diff_z[1] = normalize_angle(diff_z[1])
            mean += np.dot(K, diff_z)
            cov -= np.dot(np.dot(K, H), cov)

        return mean, cov

    def planning_prediction(self, mean, cov, ctrl, R, Q):
        # ekf predict
        predict_mean = mean.copy()
        predict_cov = cov.copy()
        predict_mean, predict_cov = self.ekf_slam_prediction(predict_mean, predict_cov, ctrl, R)

        # predict observation using maximum likelihood
        observations = []
        agent_mean = predict_mean[0: self.nStates]
        for id in range(self.nLandmark):
            if self.observed_landmarks[id] == 1:
                landmark_mean = predict_mean[2 + 2 * id + 1 : 2 + 2 * id + 2 + 1]
                dist = sqrt((agent_mean[0] - landmark_mean[0])**2 + (agent_mean[1] - landmark_mean[1])**2)
                if dist < self.sensor_range:
                    predict_observation = self.range_bearing(id, agent_mean, landmark_mean)
                    observations.append(predict_observation)
            else: # landmark hasn't been observed yet
                pass

        # ekf correct
        predict_mean, predict_cov = self.ekf_correction(predict_mean, predict_cov, observations, Q)

        # return
        return predict_mean, predict_cov

    def generate_ellipse(self, x, y, theta, a, b):
        NPOINTS = 200
        # compose point vector
        ivec = np.arange(0, 2 * pi, 2 * pi / NPOINTS)
        p = np.zeros((2, NPOINTS))
        p[0, :] = a * cos(ivec)
        p[1, :] = b * sin(ivec)

        # translate and rotate
        R = np.array([
            [cos(theta), -sin(theta)],
            [sin(theta), cos(theta)]
        ])
        p = np.dot(R, p)
        p[0, :] += x
        p[1, :] += y

        return p

    def generate_cov_ellipse(self, mean, cov, alpha=1):
        sxx = cov[0, 0]
        syy = cov[1, 1]
        sxy = cov[0, 1]
        a = alpha * np.sqrt(0.5 * (sxx + syy + np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2)))
        b = alpha * np.sqrt(0.5 * (sxx + syy - np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2)))
        theta = mean[2]  # % (2*pi)

        p = self.generate_ellipse(mean[0], mean[1], theta, a, b)
        return p

    def generate_landmark_ellipse(self, mean, cov, alpha=1):
        sxx = cov[0, 0]
        syy = cov[1, 1]
        sxy = cov[0, 1]
        a = alpha * np.sqrt(0.5 * (sxx + syy + np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2)))
        b = alpha * np.sqrt(0.5 * (sxx + syy - np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2)))
        x = mean[0]
        y = mean[1]

        NPOINTS = 200
        # compose point vector
        ivec = np.arange(0, 2 * pi, 2 * pi / NPOINTS)
        p = np.zeros((2, NPOINTS))
        p[0, :] = a * cos(ivec)
        p[1, :] = b * sin(ivec)

        # translate
        p[0, :] += x
        p[1, :] += y

        return p

    def plot(self, point_size=1, save=None):
        # [xy, vals] = self.init_t_dist.get_grid_spec()
        # plt.contourf(*xy, vals, levels=20)

        for i in range(self.landmarks.shape[0]):
            if self.observed_landmarks[i] == 1:
                plt.scatter(self.landmarks[i, 0], self.landmarks[i, 1],
                            color='orange', marker='P')
            else:
                plt.scatter(self.landmarks[i, 0], self.landmarks[i, 1],
                    color='blue', marker='P')

        xt_true = np.stack(self.log['trajectory_true'])
        traj_true = plt.scatter(xt_true[:self.tf, 0], xt_true[:self.tf, 1], s=point_size, c='red')
        # xt_dr = np.stack(self.log['trajectory_dr'])
        # traj_dr = plt.scatter(xt_dr[:self.tf, 0], xt_dr[:self.tf, 1], s=point_size, c='cyan')
        xt_est = np.stack(self.log['mean'])
        traj_est = plt.scatter(xt_est[:self.tf, 0], xt_est[:self.tf, 1], s=point_size, c='green')

        # plt.legend([traj_true, traj_dr, traj_est], ['True Path', 'Dead Reckoning Path', 'Estimated Path'])
        plt.legend([traj_true, traj_est], ['True Path', 'Estimated Path'])

        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        if save is not None:
            plt.savefig(save)
        plt.show()
        # return plt.gcf()

    def animate(self, point_size=1, show_traj=True, plan=False, save=None, rate=50, title='Animation'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_title(title)

        ax.scatter(self.landmarks[:,0], self.landmarks[:,1], color='black', marker='P')

        xt_true = np.stack(self.log['trajectory_true'])
        np.save('xt_true.npy', xt_true)
        points_true = ax.scatter([], [], s=point_size, color='red')
        agent_true = ax.scatter([], [], s=point_size * 100, color='red', marker='8')

        # xt_dr = np.stack(self.log['trajectory_dr'])
        # points_dr = ax.scatter([], [], s=point_size, c='cyan')

        mean_est = np.stack(self.log['trajectory_slam'])
        print('mean_est.shape: ', mean_est.shape)
        xt_est = mean_est
        np.save('xt_est.npy', xt_est)
        points_est = ax.scatter([], [], s=point_size, color='green')
        agent_est = ax.scatter([], [], s=point_size * 100, color='green', marker='8')

        if plan:
            mean_plan = np.stack(self.log['planning_mean'])
            xt_plan = mean_plan[:, 0:3]
        points_plan = ax.scatter([], [], s=point_size, color='yellow')
        agent_plan = ax.scatter([], [], s=point_size * 100, color='yellow', marker='8')

        observation_lines = []
        landmark_ellipses = []
        for id in range(self.landmarks.shape[0]):
            observation_lines.append(ax.plot([], [], color='orange'))
            landmark_ellipses.append(ax.scatter([], [], s=point_size, c='blue'))

        agent_ellipse = ax.scatter([], [], s=point_size, c='green')
        agent_plan_ellipse = ax.scatter([], [], s=point_size, c='yellow')

        # plt.legend([points_true, points_dr, points_est], ['True Path', 'Dead Reckoning Path', 'Estimated Path'])
        plt.legend([agent_true, agent_est], ['True Path', 'Estimated Path'])

        sensor_points = []
        for id in range(self.landmarks.shape[0]):
            sensor_point = ax.plot([], [], color='orange')
            sensor_points.append(sensor_point)

        sim_traj = ax.scatter([], [], s=point_size, c='purple')

        def sub_animate(i):
            # for debug: save frame
            # if i == 1600:
            #     plt.savefig('frame-1600.png')

            # visualize agent location / trajectory
            if (show_traj):
                points_true.set_offsets(np.array([xt_true[:i, 0], xt_true[:i, 1]]).T)
                points_est.set_offsets(np.array([xt_est[:i, 0], xt_est[:i, 1]]).T)
                if plan:
                    points_plan.set_offsets(np.array([xt_plan[:i, 0], xt_plan[:i, 1]]).T)

                agent_true.set_offsets(np.array([[xt_true[i, 0]], [xt_true[i, 1]]]).T)
                agent_est.set_offsets(np.array([[xt_est[i, 0]], [xt_est[i, 1]]]).T)

                if plan:
                    agent_plan.set_offsets(np.array([[xt_plan[i, 0]], [xt_plan[i, 1]]]).T)
            else:
                agent_true.set_offsets(np.array([[xt_true[i, 0]], [xt_true[i, 1]]]).T)
                agent_est.set_offsets(np.array([[xt_est[i, 0]], [xt_est[i, 1]]]).T)

                if plan:
                    agent_plan.set_offsets(np.array([[xt_plan[i, 0]], [xt_plan[i, 1]]]).T)

            # visualize agent covariance matrix as ellipse
            mean = self.log['mean'][i]
            agent_mean = mean[0:self.nStates]
            cov = self.log['covariance'][i]
            agent_cov = cov[0:self.nStates - 1, 0:self.nStates - 1]
            p_agent = self.generate_cov_ellipse(agent_mean, agent_cov, alpha=1)
            agent_ellipse.set_offsets(np.array([p_agent[0, :], p_agent[1, :]]).T)

            # visualize predicted planning covariance ellipse
            if plan:
                planned_mean = self.log['planning_mean'][i]
                planned_agent_mean = planned_mean[0:self.nStates]
                planned_cov = self.log['planning_cov'][i]
                planned_agent_cov = planned_cov[0:self.nStates - 1, 0:self.nStates - 1]
                planned_p_agent = self.generate_cov_ellipse(planned_agent_mean, planned_agent_cov, alpha=1)
                agent_plan_ellipse.set_offsets(np.array([planned_p_agent[0, :], planned_p_agent[1, :]]).T)

            # visualize landmark mean and covariance
            for id in range(self.nLandmark):
                landmark_ellipses[id].set_offsets(np.array([[], []]).T)

            num_lm = int((mean.shape[0]-3)/2)
            for id in range(num_lm):
                landmark_mean = self.log['mean'][i][2 + 2 * id + 1: 2 + 2 * id + 2 + 1]
                landmark_cov = self.log['covariance'][i][2 + 2 * id + 1: 2 + 2 * id + 2 + 1, 2 + 2 * id + 1: 2 + 2 * id + 2 + 1]
                # print('landmark_cov: ', landmark_cov)
                p_landmark = self.generate_landmark_ellipse(landmark_mean, landmark_cov)
                landmark_ellipses[id].set_offsets(np.array([p_landmark[0, :], p_landmark[1, :]]).T)

            # clear observation model visualization
            for point in sensor_points:
                point[0].set_xdata([])
                point[0].set_ydata([])

            # observation model visualization
            id = 0
            for obsv in self.log['observations'][i]:
                observation = obsv
                lm = self.observe_landmark(agent_mean, observation)
                sensor_points[id][0].set_xdata([xt_true[i, 0], lm[0]])
                sensor_points[id][0].set_ydata([xt_true[i, 1], lm[1]])
                id += 1

            if i<self.switch:
                sim_traj.set_offsets([-1., -1.])
            else:
                mpc_ctrls = self.log['erg_ctrls'][i-self.switch]
                mpc_traj = [xt_est[i][0:3]]
                for k in range(self.horizon):
                    ctrl = mpc_ctrls[k]
                    state = mpc_traj[k].copy()
                    state[0] += cos(state[2]) * ctrl[0] * 0.1
                    state[1] += sin(state[2]) * ctrl[0] * 0.1
                    state[2] += ctrl[1] * 0.1
                    mpc_traj.append(state)
                mpc_traj = np.array(mpc_traj)
                sim_traj.set_offsets(mpc_traj[:, 0:2])

            # return matplotlib objects for animation
            # ret = [points_true, points_dr, agent_ellipse, points_est]
            ret = [sim_traj, points_true, agent_ellipse, points_est, agent_true, agent_est, agent_plan_ellipse, agent_plan, points_plan]
            for item in sensor_points:
                ret.append(item[0])
            for item in landmark_ellipses:
                ret.append(item)
            return ret

        anim = animation.FuncAnimation(fig, sub_animate, frames=self.tf, interval=(1000 / rate), blit=True)
        if save is not None:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=40, metadata=dict(artist='simulation_slam'), bitrate=5000)
            anim.save(save, writer=writer)
        plt.show()
        # return anim

    def animate2(self, point_size=1, alpha=1, show_traj=True, plan=False, save=None, rate=50, title='Animation'):
        fig = plt.figure()

        ax1 = fig.add_subplot(121)
        [xy, vals] = self.init_t_dist.get_grid_spec()
        # ax1.contourf(*xy, vals, levels=20)
        ax1.scatter(self.landmarks[:, 0], self.landmarks[:, 1], color='white', marker='P')
        ax1.set_aspect('equal', 'box')
        ax1.set_title(title)
        ax1.set_xlim(0, self.size)
        ax1.set_ylim(0, self.size)

        ax3 = fig.add_subplot(122)
        ax3.set_aspect('equal', 'box')
        ax3.set_title('Target Distribution')

        xt_true = np.stack(self.log['trajectory_true'])
        points_true = ax1.scatter([], [], s=point_size, color='red')
        agent_true = ax1.scatter([], [], s=point_size * 100, color='red', marker='8')

        # xt_dr = np.stack(self.log['trajectory_dr'])
        # points_dr = ax1.scatter([], [], s=point_size, c='cyan')

        # mean_est = np.stack(self.log['mean'])
        mean_est = np.stack(self.log['trajectory_slam'])
        xt_est = mean_est.copy()
        points_est = ax1.scatter([], [], s=point_size, color='green')
        agent_est = ax1.scatter([], [], s=point_size * 100, color='green', marker='8')

        sim_traj = ax1.scatter([], [], s=point_size, c='purple')

        observation_lines = []
        landmark_ellipses = []
        for id in range(self.landmarks.shape[0]):
            observation_lines.append(ax1.plot([], [], color='orange'))
            landmark_ellipses.append(ax1.scatter([], [], s=point_size, c='blue'))

        agent_ellipse = ax1.scatter([], [], s=point_size, c='green')

        ax1.legend([agent_true, agent_est], ['True Path', 'Estimated Path'])

        annot = []
        for i in range(self.landmarks.shape[0]):
            annot.append(ax1.annotate('', [0.5, 0.5], size=10))

        sensor_points = []
        for id in range(self.landmarks.shape[0]):
            sensor_point = ax1.plot([], [], color='orange')
            sensor_points.append(sensor_point)

        self.ax3_cb = None

        temp_grid = np.meshgrid(*[np.linspace(0, self.size, self.num_pts+1) for _ in range(2)])
        cmap = plt.get_cmap('hot')

        def sub_animate(i):
            # visualize agent location / trajectory
            if (show_traj):
                points_true.set_offsets(np.array([xt_true[:i, 0], xt_true[:i, 1]]).T)
                points_est.set_offsets(np.array([xt_est[:i, 0], xt_est[:i, 1]]).T)
                agent_est.set_offsets(np.array([[xt_est[i, 0]], [xt_est[i, 1]]]).T)
                agent_true.set_offsets(np.array([[xt_true[i, 0]], [xt_true[i, 1]]]).T)
            else:
                agent_true.set_offsets(np.array([[xt_true[i, 0]], [xt_true[i, 1]]]).T)
                agent_est.set_offsets(np.array([[xt_est[i, 0]], [xt_est[i, 1]]]).T)

            # visualize agent covariance matrix as ellipse
            mean = self.log['mean'][i]
            agent_mean = mean[0:self.nStates]
            cov = self.log['covariance'][i]
            agent_cov = cov[0:self.nStates - 1, 0:self.nStates - 1]
            p_agent = self.generate_cov_ellipse(agent_mean, agent_cov, alpha=alpha)
            agent_ellipse.set_offsets(np.array([p_agent[0, :], p_agent[1, :]]).T)

            # visualize landmark mean and covariance
            for id in range(self.nLandmark):
                landmark_ellipses[id].set_offsets(np.array([[], []]).T)

            for id in range(int((mean.shape[0]-3)/2)):
                if mean[2 + 2 * id + 1] == 0:
                    annot[id].set_text('')
                else:
                    landmark_mean = mean[2 + 2 * id + 1: 2 + 2 * id + 2 + 1]
                    landmark_cov = cov[2 + 2 * id + 1: 2 + 2 * id + 2 + 1, 2 + 2 * id + 1: 2 + 2 * id + 2 + 1]
                    p_landmark = self.generate_landmark_ellipse(landmark_mean, landmark_cov)
                    landmark_ellipses[id].set_offsets(np.array([p_landmark[0, :], p_landmark[1, :]]).T)
                    annot[id].set_text("{:.2E}".format(np.linalg.det(landmark_cov)))
                    annot[id].set_x(landmark_mean[0])
                    annot[id].set_y(landmark_mean[1])

            # clear observation model visualization
            for point in sensor_points:
                point[0].set_xdata([])
                point[0].set_ydata([])

            # observation model visualization
            idx = 0
            for obsv in self.log['observations'][i]:
                observation = obsv
                lm = self.observe_landmark(agent_mean, observation)
                sensor_points[idx][0].set_xdata([xt_true[i, 0], lm[0]])
                sensor_points[idx][0].set_ydata([xt_true[i, 1], lm[1]])
                idx += 1

            # visualize true path statistics
            path_true = xt_true[:i+1, self.model_true.explr_idx]
            ck_true = convert_traj2ck(self.erg_ctrl_true.basis, path_true)
            val_true = convert_ck2dist(self.erg_ctrl_true.basis, ck_true, size=self.size)

            # visualize planning trajectory
            if i<self.switch:
                sim_traj.set_offsets([-1., -1.])
            else:
                mpc_ctrls = self.log['erg_ctrls'][i-self.switch]
                mpc_traj = [xt_est[i][0:3]]
                for k in range(self.horizon):
                    ctrl = mpc_ctrls[k]
                    state = mpc_traj[k].copy()
                    state[0] += cos(state[2]) * ctrl[0] * 0.1
                    state[1] += sin(state[2]) * ctrl[0] * 0.1
                    state[2] += ctrl[1] * 0.1
                    mpc_traj.append(state)
                mpc_traj = np.array(mpc_traj)
                sim_traj.set_offsets(mpc_traj[:, 0:2])

            # visualize target distribution
            t_dist = self.log['target_dist'][i]
            xy3, vals = t_dist.get_grid_spec()
            ax3.cla()
            ax3.set_title('Target Distribution')
            ax3_countour = ax3.contourf(*xy3, vals, levels=20, cmap=cmap)
            # ax3_grid = ax3.pcolormesh(temp_grid[0], temp_grid[1], vals, cmap=cmap)

            # return matplotlib objects for animation
            ret = [points_true, agent_ellipse, points_est, agent_true, agent_est]
            for item in sensor_points:
                ret.append(item[0])
            for item in landmark_ellipses:
                ret.append(item)
            for item in annot:
                ret.append(item)
            return ret

        anim = animation.FuncAnimation(fig, sub_animate, frames=self.tf, interval=(1000 / rate))
        if save is not None:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=40, metadata=dict(artist='simulation_slam'), bitrate=5000)
            anim.save(save, writer=writer)
        plt.show()
        # return anim


    def path_reconstruct(self, save=None):
        plt.clf()
        plt.close()
        fig = plt.figure()

        ax1 = fig.add_subplot(221)
        ax1.set_aspect('equal', 'box')
        # ax1.scatter(self.landmarks[:, 0], self.landmarks[:, 1],
        #             color='blue', marker='P')
        ax1.set_title('Landmark Distribution')
        ax1.set_xlim(0, self.size)
        ax1.set_ylim(0, self.size)
        #################################################

        for i in range(self.landmarks.shape[0]):
            if self.observed_landmarks[i] == 1:
                ax1.scatter(self.landmarks[i, 0], self.landmarks[i, 1],
                            color='orange', marker='P')
            else:
                ax1.scatter(self.landmarks[i, 0], self.landmarks[i, 1],
                    color='blue', marker='P')

        xt_true = np.stack(self.log['trajectory_true'])
        traj_true = ax1.scatter(xt_true[:self.tf, 0], xt_true[:self.tf, 1], s=1, c='red')
        xt_est = np.stack(self.log['mean'])
        traj_est = ax1.scatter(xt_est[:self.tf, 0], xt_est[:self.tf, 1], s=1, c='green')

        ax1.legend([traj_true, traj_est], ['True Path', 'Estimated Path'])

        #################################################
        ax2 = fig.add_subplot(222)
        ax2.set_aspect('equal', 'box')
        t_dist = self.log['target_dist'][-1]
        xy, vals = t_dist.get_grid_spec()
        ax2.set_title('Final Target Distribution')
        ax2_countour = ax2.contourf(*xy, vals, levels=25)

        xy, vals = self.t_dist.get_grid_spec()

        path_true = np.stack(self.log['trajectory_true'])[:self.tf, self.model_true.explr_idx]
        ck_true = convert_traj2ck(self.erg_ctrl_true.basis, path_true)
        val_true = convert_ck2dist(self.erg_ctrl_true.basis, ck_true, size=self.size)

        # path_dr = np.stack(self.log['trajectory_dr'])[:self.tf, self.model_dr.explr_idx]
        # ck_dr = convert_traj2ck(self.erg_ctrl_dr.basis, path_dr)
        # val_dr = convert_ck2dist(self.erg_ctrl_dr.basis, ck_dr, size=self.size)

        path_est = np.stack(self.log['mean'])[:self.tf, self.model_dr.explr_idx]
        ck_est = convert_traj2ck(self.erg_ctrl_dr.basis, path_est)
        val_est = convert_ck2dist(self.erg_ctrl_dr.basis, ck_est, size=self.size)

        ax3 = fig.add_subplot(223)
        ax3.contourf(*xy, val_true.reshape(50, 50), levels=20)
        ax3.set_aspect('equal', 'box')
        ax3.set_title('Actual Path Statistics')

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

