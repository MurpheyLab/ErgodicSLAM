import matplotlib.pyplot as plt
from matplotlib import animation
import autograd.numpy as np
from .utils import convert_ck2dist, convert_traj2ck, normalize_angle
from tqdm import tqdm
from numpy import sin, cos, sqrt
import math
from math import pi
from tempfile import TemporaryFile
from scipy.optimize import minimize
from time import time


class simulation_slam():
    def __init__(self, size, init_state, t_dist, model_true, erg_ctrl_true, env_true, model_dr, erg_ctrl_dr, env_dr, tf,
                 landmarks, sensor_range, motion_noise, measure_noise, static_test=None):
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
        self.measure_noise = measure_noise
        self.Q = np.diag(measure_noise) ** 2
        self.observed_landmarks = np.zeros(self.landmarks.shape[0])
        self.curr_obsv = []
        # self.threshold = 99999999
        self.threshold = 99999999.

        self.static_test = static_test

    def start(self, report=False, debug=False):
        #########################
        # initialize mean and covariance matrix
        #########################
        self.nLandmark = self.landmarks.shape[0]  # number of landmarks
        self.nStates = self.init_state.shape[0]
        self.dim = self.nStates + 2 * self.nLandmark
        mean = np.zeros(self.dim)
        cov = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            if i < self.init_state.shape[0]:
                mean[i] = self.init_state[i]
                cov[i, i] = 0
            else:
                cov[i, i] = self.threshold

        ##########################
        # simulation loop
        ##########################
        self.log = {'trajectory_true': [], 'trajectory_dr': [], 'true_landmarks': [], 'observations': [], 'mean': [],
                    'covariance': [], 'planning_mean': [], 'planning_cov': []}
        state_true = self.env_true.reset(self.init_state)
        state_dr = self.env_dr.reset(self.init_state)

        self.curr_t = 0
        for t in tqdm(range(self.tf)):
            self.curr_t = t
            print('cov: ', cov)
            #########################
            # generate control and measurement data
            #########################
            # this is what robot thinks it's doing
            if debug:  # debug mode: robot runs a circle
                ctrl = np.array([2.0, 0.5])
            else:
                ctrl = self.erg_ctrl_dr(mean[0:self.nStates])
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
                    noisy_observation = self.range_bearing(i, state_true, item)
                    noisy_observation[1:] += self.measure_noise * np.random.randn(2)
                    observations.append(noisy_observation)
                    self.curr_obsv.append(i)
            self.log['true_landmarks'].append(true_landmarks)
            self.log['observations'].append(np.array(observations))

            ########################
            # Planning prediction
            ########################

            planning_predict_mean, planning_predict_cov = self.planning_prediction(mean, cov, ctrl, self.R, self.Q)
            planning_predict_mean[2] = normalize_angle(planning_predict_mean[2])
            self.log['planning_mean'].append(planning_predict_mean)
            self.log['planning_cov'].append(planning_predict_cov)
            print('mpc mean:\n', planning_predict_mean[0:3])

            #########################
            # MPC Planning Test
            #########################
            if self.static_test is not None:
                if self.static_test == t:
                    self.mpc_planning(mean, cov, np.array([0., 0.]))
                else:
                    pass
            else:
                if self.static_test == 'all':
                    self.mpc_planning(mean, cov, np.array([0., 0.]))
                else:
                    pass

            #########################
            # EKF SLAM
            #########################
            mean[2] = normalize_angle(mean[2])
            mean, cov = self.ekf_prediction(mean, cov, ctrl, self.R)
            mean[2] = normalize_angle(mean[2])

            mean,cov = self.ekf_correction(mean,cov, observations, self.Q)
            mean[2] = normalize_angle(mean[2])
            print('est mean:', mean[0:3])

            self.log['mean'].append(mean)
            self.log['covariance'].append(cov)

            # mean = predict_mean
            # cov = predict_cov
            # print("current observation: ", self.curr_obsv)

            #########################
            # Information for debug
            #########################
            # print("robot uncertainty: ", np.linalg.det(cov[0: self.nStates, 0: self.nStates]))
            # if t == 1600:
            #     np.save("/home/msun/Code/ErgodicBSP/test/mean_mat", mean)
            #     np.save("/home/msun/Code/ErgodicBSP/test/cov_mat", cov)
            #     print("data written success !")

        print("simulation finished.")

    def range_bearing(self, id, agent, landmark):
        delta = landmark - agent[0:self.nStates - 1]
        range = np.sqrt(np.dot(delta.T, delta))
        bearing = math.atan2(delta[1], delta[0]) - agent[2]
        bearing = normalize_angle(bearing)
        return np.array([id, range, bearing])

    def ekf_prediction(self, mean, cov, ctrl, R):
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

    def ekf_correction_mpc(self, predict_mean, predict_cov, z, Q):
        # initialize mean and cov
        mean = predict_mean.copy()
        cov = predict_cov.copy()

        # iterate each observed landmark
        for obs in z:
            # extract measurement data
            id = int(obs[0])
            measurement = obs[1:]
            # est_landmark = self.landmarks[id]
            est_landmark = np.array([
                    mean[0] + measurement[0] * cos(measurement[1]+mean[2]),
                    mean[1] + measurement[0] * sin(measurement[1]+mean[2])
                ])
            # assume perfect observation,
            #   so expected measurement is same as the input
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
            print('H: ', H)
            print('K: ', K)
            # update mean and covariance matrix
            diff_z = measurement - zi
            diff_z[1] = normalize_angle(diff_z[1])
            mean += np.dot(K, diff_z)
            cov -= np.dot(np.dot(K, H), cov)

        return mean, cov

    def mpc_planning(self, mean, cov, ctrl, horizon=1):
        y_init = [np.concatenate((mean.reshape(-1), cov.reshape(-1), ctrl.reshape(-1)))]
        mean_init = mean.copy()
        cov_init = cov.copy()
        for i in range(horizon):
            mean_init, cov_init = self.planning_prediction(mean_init, cov_init, ctrl, self.R, self.Q)
            y_init.append(np.concatenate((mean_init.reshape(-1), cov_init.reshape(-1), ctrl.reshape(-1))))
        y_init = np.array(y_init).reshape(-1)

        cons = []
        cons.append({'type':'eq', 'fun':lambda y:self.mpc_constraint(y, horizon)})
        cons.append({'type':'eq', 'fun':lambda y:self.mpc_init_cond(y, y_init, horizon)})
        objective = lambda y:self.mpc_objective(y, horizon)
        res = minimize(objective, y_init, method='SLSQP', constraints=cons, options={'maxiter':1000, 'disp':True})
        y_res = res.x.reshape(horizon+1, -1)
        np.save('static_mean_t{}.npy'.format(self.curr_t), mean)
        np.save('static_cov_t{}.npy'.format(self.curr_t), cov)
        np.save('static_y_res_t{}.npy'.format(self.curr_t), y_res)
        self.y_res = y_res

    def mpc_init_cond(self, y, y_init, horizon):
        new_y = y.reshape(horizon+1, -1)
        new_x = new_y[0]
        new_y_init = y_init.reshape(horizon+1, -1)
        new_x_init = new_y_init[0]
        return new_x - new_x_init

    def mpc_constraint(self, y, horizon):
        cons = []
        new_y = y.reshape(horizon+1, -1)
        for i in range(horizon):
            xt = new_y[i]
            mean = xt[0:self.dim]
            cov = xt[self.dim:-2].reshape(self.dim, self.dim)
            ctrl = xt[-2:]
            mean_tt, cov_tt = self.planning_prediction(mean, cov, ctrl, self.R, self.Q)
            xtt = np.concatenate((mean_tt.reshape(-1), cov_tt.reshape(-1), ctrl.reshape(-1)))
            diff = new_y[i+1] - xtt
            for item in diff:
                cons.append(item)
        return np.array(cons)

    def mpc_objective(self, y, horizon):
        obj = 0
        new_y = y.reshape(horizon+1, -1)
        for state in new_y:
            cov_flat = state[self.dim:-2]
            cov = cov_flat.reshape(self.dim, self.dim)
            obj += np.trace(cov)
        return obj

    # mpc-based implementaton of ekf-ml prediction,
    #   assume all landmarks observed at last time step
    #	can be observed in the horizon (ensure continuity)
    # '''
    def planning_prediction(self, mean, cov, ctrl, R, Q):
        start = time()
        # ekf predict
        predict_mean = mean.copy()
        predict_cov = cov.copy()
        predict_mean, predict_cov = self.ekf_prediction(predict_mean, predict_cov, ctrl, R)

        # ekf correction (assume perfect observation)
        predict_obsv = []
        for i in self.curr_obsv:
            lm = self.landmarks[i]
            predict_obsv.append(self.range_bearing(i, predict_mean[0:3], lm))
        predict_mean, predict_cov = self.ekf_correction_mpc(predict_mean, predict_cov, predict_obsv, Q)

        print("elapsed time: ", time()-start)

        # return
        return predict_mean, predict_cov
    # '''

    # old implementaton of ekf-ml prediction,
    #   has discontinuity for observation
    '''
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
    '''

    def generate_ellipse(self, x, y, theta, a, b):
        NPOINTS = 100
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

        NPOINTS = 100
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
        [xy, vals] = self.t_dist.get_grid_spec()
        plt.contourf(*xy, vals, levels=20)

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
        if save is not None:
            plt.savefig(save)
        plt.show()
        # return plt.gcf()

    def animate(self, point_size=1, show_traj=True, plan=False, save=None, rate=50, title='Animation'):
        [xy, vals] = self.t_dist.get_grid_spec()
        plt.contourf(*xy, vals, levels=20)
        plt.scatter(self.landmarks[:, 0], self.landmarks[:, 1], color='white', marker='P')
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        ax.set_title(title)
        fig = plt.gcf()

        xt_true = np.stack(self.log['trajectory_true'])
        points_true = ax.scatter([], [], s=point_size, color='red')
        agent_true = ax.scatter([], [], s=point_size * 100, color='red', marker='8')

        # xt_dr = np.stack(self.log['trajectory_dr'])
        # points_dr = ax.scatter([], [], s=point_size, c='cyan')

        mean_est = np.stack(self.log['mean'])
        xt_est = mean_est[:, 0:3]
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
            landmark_ellipses.append(ax.scatter([], [], s=point_size, c='cyan'))

        agent_ellipse = ax.scatter([], [], s=point_size, c='green')
        agent_plan_ellipse = ax.scatter([], [], s=point_size, c='yellow')

        # plt.legend([points_true, points_dr, points_est], ['True Path', 'Dead Reckoning Path', 'Estimated Path'])
        plt.legend([agent_true, agent_est], ['True Path', 'Estimated Path'])

        sensor_points = []
        for id in range(self.landmarks.shape[0]):
            sensor_point = ax.plot([], [], color='orange')
            sensor_points.append(sensor_point)

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

            for id in range(self.nLandmark):
                if mean[2 + 2 * id + 1] == 0:
                    pass
                else:
                    landmark_mean = mean[2 + 2 * id + 1: 2 + 2 * id + 2 + 1]
                    landmark_cov = cov[2 + 2 * id + 1: 2 + 2 * id + 2 + 1, 2 + 2 * id + 1: 2 + 2 * id + 2 + 1]
                    p_landmark = self.generate_landmark_ellipse(landmark_mean, landmark_cov)
                    landmark_ellipses[id].set_offsets(np.array([p_landmark[0, :], p_landmark[1, :]]).T)

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
            # ret = [points_true, points_dr, agent_ellipse, points_est]
            ret = [points_true, agent_ellipse, points_est, agent_true, agent_est, agent_plan_ellipse, agent_plan, points_plan]
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

    def animate3(self, point_size=1, show_traj=True, plan=False, save=None, rate=50, title='Animation'):
        fig = plt.gcf()

        ax1 = fig.add_subplot(131)
        [xy, vals] = self.t_dist.get_grid_spec()
        ax1.contourf(*xy, vals, levels=20)
        ax1.scatter(self.landmarks[:, 0], self.landmarks[:, 1], color='white', marker='P')
        ax1.set_aspect('equal', 'box')
        ax1.set_title(title)

        ax2 = fig.add_subplot(132)
        ax2.set_aspect('equal', 'box')
        ax2.set_title('Actual Path Statistics')

        ax3 = fig.add_subplot(133)
        ax3.set_aspect('equal', 'box')
        ax3.set_title('Estimated Path Statistics')

        xt_true = np.stack(self.log['trajectory_true'])
        points_true = ax1.scatter([], [], s=point_size, color='red')
        agent_true = ax1.scatter([], [], s=point_size * 100, color='red', marker='8')

        # xt_dr = np.stack(self.log['trajectory_dr'])
        # points_dr = ax1.scatter([], [], s=point_size, c='cyan')

        mean_est = np.stack(self.log['mean'])
        xt_est = mean_est[:, 0:3]
        points_est = ax1.scatter([], [], s=point_size, color='green')
        agent_est = ax1.scatter([], [], s=point_size * 100, color='green', marker='8')

        mean_plan = np.stack(self.log['planning_mean'])
        xt_plan = mean_plan[:, 0:3]
        points_plan = ax1.scatter([], [], s=point_size, color='yellow')
        agent_plan = ax1.scatter([], [], s=point_size * 100, color='yellow', marker='8')

        observation_lines = []
        landmark_ellipses = []
        for id in range(self.landmarks.shape[0]):
            observation_lines.append(ax1.plot([], [], color='orange'))
            landmark_ellipses.append(ax1.scatter([], [], s=point_size, c='cyan'))

        agent_ellipse = ax1.scatter([], [], s=point_size, c='green')
        agent_plan_ellipse = ax1.scatter([], [], s=point_size, c='yellow')

        # ax1.legend([points_true, points_dr, points_est], ['True Path', 'Dead Reckoning Path', 'Estimated Path'])
        ax1.legend([agent_true, agent_est], ['True Path', 'Estimated Path'])

        sensor_points = []
        for id in range(self.landmarks.shape[0]):
            sensor_point = ax1.plot([], [], color='orange')
            sensor_points.append(sensor_point)

        def sub_animate(i):
            # visualize agent location / trajectory
            if (show_traj):
                points_true.set_offsets(np.array([xt_true[:i, 0], xt_true[:i, 1]]).T)
                points_est.set_offsets(np.array([xt_est[:i, 0], xt_est[:i, 1]]).T)
                # points_plan.set_offsets(np.array([xt_plan[:i, 0], xt_plan[:i, 1]]).T)

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

            for id in range(self.nLandmark):
                if mean[2 + 2 * id + 1] == 0:
                    pass
                else:
                    landmark_mean = mean[2 + 2 * id + 1: 2 + 2 * id + 2 + 1]
                    landmark_cov = cov[2 + 2 * id + 1: 2 + 2 * id + 2 + 1, 2 + 2 * id + 1: 2 + 2 * id + 2 + 1]
                    p_landmark = self.generate_landmark_ellipse(landmark_mean, landmark_cov)
                    landmark_ellipses[id].set_offsets(np.array([p_landmark[0, :], p_landmark[1, :]]).T)

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

            # visualize path statistics
            path_true = xt_true[:i+1, self.model_true.explr_idx]
            ck_true = convert_traj2ck(self.erg_ctrl_true.basis, path_true)
            val_true = convert_ck2dist(self.erg_ctrl_true.basis, ck_true, size=self.size)
            ax2.cla()
            ax2.contourf(*xy, val_true.reshape(50, 50), levels=20)

            # visualize path statistics
            path_est = xt_est[:i+1, self.model_dr.explr_idx]
            ck_est = convert_traj2ck(self.erg_ctrl_dr.basis, path_est)
            val_est = convert_ck2dist(self.erg_ctrl_dr.basis, ck_est, size=self.size)
            ax3.cla()
            ax3.contourf(*xy, val_est.reshape(50, 50), levels=20)

            # return matplotlib objects for animation
            # ret = [points_true, points_dr, agent_ellipse, points_est]
            ret = [points_true, agent_ellipse, points_est, agent_true, agent_est, agent_plan_ellipse, agent_plan, points_plan]
            for item in sensor_points:
                ret.append(item[0])
            for item in landmark_ellipses:
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

        plt.cla()
        plt.clf()
        plt.close()
        fig = plt.figure()

        ax1 = fig.add_subplot(131)
        ax1.contourf(*xy, vals, levels=20)
        ax1.set_aspect('equal', 'box')
        ax1.set_title('Spatial Distribution')

        ax2 = fig.add_subplot(132)
        ax2.contourf(*xy, val_true.reshape(50, 50), levels=20)
        ax2.set_aspect('equal', 'box')
        ax2.set_title('Actual Path Statistics')

        # ax3 = fig.add_subplot(223)
        # ax3.contourf(*xy, val_dr.reshape(50, 50), levels=20)
        # ax3.set_aspect('equal', 'box')
        # ax3.set_title('Dead Reckoning Path Statistics')

        ax4 = fig.add_subplot(133)
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

    def static_test_plot(self, point_size=1, save=None):
        if self.static_test is None:
            return -1

        # plot origin traj
        [xy, vals] = self.t_dist.get_grid_spec()
        plt.contourf(*xy, vals, levels=20)

        xt_true = np.stack(self.log['trajectory_true'])
        traj_true = plt.scatter(xt_true[:self.static_test, 0], xt_true[:self.static_test, 1], s=point_size, c='red')
        xt_est = np.stack(self.log['mean'])
        traj_est = plt.scatter(xt_est[:self.static_test, 0], xt_est[:self.static_test, 1], s=point_size, c='green')

        plt.legend([traj_true, traj_est], ['True Path', 'Estimated Path'])

        # deal with self.y_res
        mpc_xt = self.y_res[:, 0:2]
        plt.scatter(mpc_xt[:,0], mpc_xt[:,1], s=point_size, c='yellow')

        # plot
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        if save is not None:
            plt.savefig(save)
        plt.show()
        # return plt.gcf()
