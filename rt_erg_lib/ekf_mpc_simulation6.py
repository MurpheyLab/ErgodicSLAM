"""
based on continuous objective version
add attractor
"""

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import autograd.numpy as np
from .utils import convert_ck2dist, convert_traj2ck, normalize_angle
from tqdm import tqdm
from numpy import sin, cos, sqrt
import math
from math import pi
from tempfile import TemporaryFile
from scipy.optimize import minimize
from scipy.linalg import block_diag


class simulation_slam():
    def __init__(self, size, init_state, t_dist, model_true, erg_ctrl_true, env_true, model_dr, erg_ctrl_dr, env_dr, tf,
                 landmarks, sensor_range, motion_noise, measure_noise, static_test, horizon, switch, num_pts):
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
        self.threshold = 0.

        self.lm_id = []
        self.obsv_lm = []
        self.static_test = static_test
        self.horizon = horizon
        self.switch = switch
        self.attractor = None

        self.num_pts = num_pts
        self.og_vals = np.ones(self.num_pts * self.num_pts) * 0.5
        self.raw_grid = np.meshgrid(*[np.linspace(0, self.size, self.num_pts+1) for _ in range(2)])
        self.grid = [self.raw_grid[0][0:self.num_pts,0:self.num_pts], self.raw_grid[1][0:self.num_pts,0:self.num_pts]]
        self.grid2 = np.c_[self.grid[0].ravel(), self.grid[1].ravel()]
        self.grid_x = self.grid2[:,0]
        self.grid_y = self.grid2[:,1]
        self.diff_x = self.grid_x - self.grid_x[:,np.newaxis]
        self.diff_y = self.grid_y - self.grid_y[:,np.newaxis]
        self.dist_xy = np.sqrt(self.diff_x**2 + self.diff_y**2)
        self.dist_flag = (self.dist_xy < self.sensor_range).astype(int)

    def start(self, report=False, debug=False):
        #########################
        # initialize mean and covariance matrix
        #########################
        self.nLandmark = self.landmarks.shape[0]  # number of landmarks
        self.nStates = self.init_state.shape[0]
        self.dim = self.nStates + 2 * self.nLandmark

        mean = np.zeros(self.nStates)
        cov = np.zeros((self.nStates, self.nStates))
        mean[0:3] = self.init_state

        obsv_lm = []

        ##########################
        # simulation loop
        ##########################
        self.log = {'trajectory_true': [], 'trajectory_dr': [], 'true_landmarks': [], 'observations': [], 'mean': [], 'trajectory_slam': [],
                'covariance': [], 'planning_mean': [], 'planning_cov': [], 'mpc_ctrls': [], 'og_vals':[], 'attractor':[]}
        state_true = self.env_true.reset(self.init_state)
        state_dr = self.env_dr.reset(self.init_state)

        self.curr_t = 0
        for t in tqdm(range(self.tf)):
            self.curr_t = t
            #########################
            # generate control and measurement data
            #########################
            # this is what robot thinks it's doing
            if t < self.switch: # default control
                ctrl = np.array([2.8, 0.4])
            else:
                obsv_lm = np.array([self.lm_id.index(item) for item in self.curr_obsv])
                mpc_ctrls = self.mpc_planning(mean, cov, ctrl, horizon=self.horizon, obsv_lm=obsv_lm, attractor=self.attractor)
                ctrl = mpc_ctrls[0]
                self.log['mpc_ctrls'].append(mpc_ctrls)
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
#                        cov = np.block([
#                                [cov, np.zeros((cov.shape[0],2))],
#                                [np.zeros((2,cov.shape[0])), np.zeros((2,2))]
#                            ])
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
            obsv_lm = np.array([self.lm_id.index(item) for item in self.curr_obsv])
            planning_predict_mean, planning_predict_cov = self.planning_prediction(mean, cov, ctrl, obsv_lm)
            planning_predict_mean[2] = normalize_angle(planning_predict_mean[2])
            self.log['planning_mean'].append(planning_predict_mean[0:3])
            self.log['planning_cov'].append(planning_predict_cov)
            # print('mpc mean:\n', planning_predict_mean[0:3])
            '''

            #########################
            # MPC Planning Test
            #########################
            '''
            if self.static_test is not None:
                if self.static_test == t:
                    self.mpc_planning(mean, cov, ctrl, horizon=20, obsv_lm=obsv_lm)
            '''

            #########################
            # EKF SLAM
            #   now we have estimation of states for (t-1),
            #   ctrl and observations at (t)
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

            # update attractor through state machine
            self.attractor = self.state_transition(mean[0:3], cov[0:3, 0:3])
            self.log['og_vals'].append(self.og_vals.copy())
            attractor_log = self.attractor
            self.log['attractor'].append(attractor_log)

            #########################
            # Information for debug
            #########################
            # print("robot uncertainty: ", np.linalg.det(cov[0: self.nStates, 0: self.nStates]))
            # if t == 1600:
            #     np.save("/home/msun/Code/ErgodicBSP/test/mean_mat", mean)
            #     np.save("/home/msun/Code/ErgodicBSP/test/cov_mat", cov)
            #     print("data written success !")

        print("simulation finished.")

    def state_transition(self, mean, cov):
        # update og
        for idx in range(self.grid2.shape[0]):
            g = self.grid2[idx]
            if np.linalg.norm(g-mean[0:2], 2) < self.sensor_range:
                self.og_vals[idx] = 1
            else:
                pass

        # determine attractor
        print('state transition: ', np.trace(cov))
        if np.trace(cov) < 0.5:
            dist_xy = np.sqrt((self.grid[0]-mean[0])**2 + (self.grid[1]-mean[1])**2)
            dist_flag = self.og_vals.reshape(self.num_pts, self.num_pts) < 1.0
            dist_flag = dist_flag.astype(int) * 10000
            dist_xy = dist_xy + dist_flag
            attractor = self.grid2[np.argmin(dist_xy)]
        else:
            attractor = None

        return attractor

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

    def mpc_planning(self, meann, covv, ctrl, horizon, obsv_lm, attractor=None):
        init_ctrls = np.array([ctrl.copy() for _ in range(horizon)]).reshape(-1)
        mean = meann.copy()
        cov = covv.copy()

        objective = lambda ctrls : self.mpc_objective(mean, cov, ctrls, horizon, obsv_lm, attractor)
        bounds = [[-2., 2.] for _ in range(horizon*2)]
        res = minimize(objective, x0=init_ctrls, method='trust-constr', bounds=bounds, options={'disp':True,})# 'gtol':1e-12, 'xtol':1e-12})
        # res = minimize(objective, init_ctrls, method='BFGS', tol=1e-10, options={'disp':True})
        controls = res.x.reshape(horizon, 2)
        return controls

    def mpc_objective(self, meann, covv, ctrls, horizon, obsv_lmm, attractor=None):
        '''
        obsv_lm contains id for landmarks being observed in mean
        '''
        mean = meann.copy()
        cov = covv.copy()
        obsv_lm = np.array(obsv_lmm)
        obj = 0.

        if attractor is None:
            pass
        else:
            robot_mean = mean[0:2]
            attractor = mean[0:2] + np.array([-1.0, 1.0])
            mean = np.concatenate((mean, attractor))
            cov = np.block([[cov, np.zeros((cov.shape[0], 2))], [np.zeros((2, cov.shape[0])), np.eye(2)*1e-12]])
            obsv_lm = np.concatenate((obsv_lmm, [ int((len(mean)-3)/2)-1 ]))

        for t in range(horizon):
            ctrl = ctrls[2*t:2*t+2]
            # obj += 50. ** np.linalg.norm(ctrl) - 1.
            ctrl_norm = np.linalg.norm(ctrl)
            G = np.eye(mean.shape[0])
            G[0][2] = -sin(mean[2]) * ctrl[0] * 0.1
            G[1][2] =  cos(mean[2]) * ctrl[0] * 0.1
            num_lm = int((mean.shape[0]-3) / 2)
            BigR = np.block([
                    [self.R, np.zeros((3, 2*num_lm))],
                    [np.zeros((2*num_lm, 3)), np.zeros((2*num_lm,2*num_lm))]
                ])
            cov = G @ cov @ G.T + BigR

            g = np.zeros(mean.shape[0])
            g[0] = cos(mean[2]) * ctrl[0]
            g[1] = sin(mean[2]) * ctrl[0]
            g[2] = ctrl[1]
            mean += g * 0.1
            mean[2] = normalize_angle(mean[2])

            num_obsv = obsv_lm.shape[0]
            H = np.zeros((num_obsv*2, mean.shape[0]))
            r = mean[0:3].copy()
            idx = -2
            for lid in obsv_lm:
                idx += 2
                lid = int(lid)
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

            BigQ = block_diag(*[self.Q for _ in range(num_obsv)])

            K = cov @ H.T @ np.linalg.inv(H @ cov @ H.T + BigQ)
            cov = cov - K @ H @ cov

        # print('cov: ', np.exp( np.log( (np.linalg.det(cov))**(1/mean.shape[0]) ) ) )# + obj * 0.00000001 )
        # print('obj: ', obj * 0.000001)

        # A-optimality
        return np.trace(cov) #+ obj * 0.00000001

        # test D-optimality: 0
        # return np.linalg.det(cov) + obj * 0.00000001

        # test D-optimality: 1
        # return np.exp( np.log( (np.linalg.det(cov))**(1/mean.shape[0]) ) ) + obj * 0.00000001

        # test D-optimality: 2
        '''
        obj = np.linalg.det(cov[0:3, 0:3])
        num_lm = int((mean.shape[0]-3) / 2)
        for i in range(num_lm):
            obj += np.linalg.det(cov[3+i*2:5+i*2, 3+i*2:5+i*2])
        return obj
        '''

        # test E-optimality
        # return 0.5 * np.log(2*np.pi*np.e)**mean.shape[0] * np.linalg.norm(cov)

    # mpc-based implementaton of ekf-ml prediction,
    #   assume all landmarks observed at last time step
    #	can be observed in the horizon (ensure continuity)
    def planning_prediction(self, meann, covv, ctrl, obsv_lm):
        mean = meann.copy()
        cov = covv.copy()

        for t in range(1):
#            ctrl = ctrls[t]
            G = np.eye(mean.shape[0])
            G[0][2] = -sin(mean[2]) * ctrl[0] * 0.1
            G[1][2] =  cos(mean[2]) * ctrl[0] * 0.1
            num_lm = int((mean.shape[0]-3) / 2)
            BigR = np.block([
                    [self.R, np.zeros((3, 2*num_lm))],
                    [np.zeros((2*num_lm, 3)), np.zeros((2*num_lm,2*num_lm))]
                ])
            cov = G @ cov @ G.T + BigR

            g = np.zeros(mean.shape[0])
            g[0] = cos(mean[2]) * ctrl[0]
            g[1] = sin(mean[2]) * ctrl[0]
            g[2] = ctrl[1]
            mean += g * 0.1
            mean[2] = normalize_angle(mean[2])

            num_obsv = obsv_lm.shape[0]
            H = np.zeros((num_obsv*2, mean.shape[0]))
            r = mean[0:3].copy()
            idx = -2
            for lid in obsv_lm:
                idx += 2
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

            BigQ = block_diag(*[self.Q for _ in range(num_obsv)])

            K = cov @ H.T @ np.linalg.inv(H @ cov @ H.T + BigQ)
            cov = cov - K @ H @ cov

        # return
        return mean, cov

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
        xt_est = np.stack(self.log['trajectory_slam'])
        xt_est = np.stack([item[0:2] for item in self.log['mean']])
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

            if i < self.switch:
                sim_traj.set_offsets([-1., -1.])
            else:
                mpc_ctrls = self.log['mpc_ctrls'][i-self.switch]
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

    def animate2(self, point_size=1, show_traj=True, plan=False, save=None, rate=50, title='Animation'):
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.set_aspect('equal')
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_title(title)

        ax2 = fig.add_subplot(122)
        ax2.set_aspect('equal')

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

        cmap2 = plt.get_cmap('gray')
        levels = MaxNLocator(nbins=50).tick_values(0.5, 1.0)
        norm = BoundaryNorm(levels, ncolors=cmap2.N, clip=True)

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

            if i < self.switch:
                sim_traj.set_offsets([-1., -1.])
            else:
                mpc_ctrls = self.log['mpc_ctrls'][i-self.switch]
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

            # visualize og
            ax2.cla()
            ax2.set_title('Occupancy Grid')
            ax2_grid = ax2.pcolormesh( self.raw_grid[0], self.raw_grid[1], self.log['og_vals'][i].reshape(self.num_pts, self.num_pts), cmap=cmap2, edgecolors='k', linewidth=0.004, norm=norm )
            print('og_vals sum: ', np.sum(self.log['og_vals'][i]))

            # return matplotlib objects for animation
            # ret = [points_true, points_dr, agent_ellipse, points_est]
            ret = [sim_traj, points_true, agent_ellipse, points_est, agent_true, agent_est, agent_plan_ellipse, agent_plan, points_plan]
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
        xt_est = np.stack(self.log['trajectory_slam'])
        traj_est = plt.scatter(xt_est[:self.static_test, 0], xt_est[:self.static_test, 1], s=point_size, c='green')

        plt.legend([traj_true, traj_est], ['True Path', 'Estimated Path'])

        # deal with self.y_res
        mpc_ut = self.y_res
        mpc_xt = [xt_est[self.static_test].copy()]
        idx = 0
        for u in mpc_ut:
            xt = mpc_xt[idx]
            new_xt = xt.copy()
            new_xt[0] += cos(xt[2]) * u[0] * 0.1
            new_xt[1] += sin(xt[2]) * u[0] * 0.1
            new_xt[2] += u[1] * 0.1
            mpc_xt.append(new_xt.copy())
            idx += 1
        mpc_xt = np.array(mpc_xt)

        plt.scatter(mpc_xt[:,0], mpc_xt[:,1], s=point_size, c='yellow')

        # plot
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        if save is not None:
            plt.savefig(save)
        plt.show()
        # return plt.gcf()
