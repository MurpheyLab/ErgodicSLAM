import numpy as np
from numpy import sin, cos, pi
from scipy.linalg import block_diag

from .basis import Basis
from .barrier import Barrier
from .replay_buffer import ReplayBuffer
from .utils import *

class RTErgodicControl(object):

    def __init__(self, model, target_dist,
                    weights=None, horizon=100, num_basis=5,
                    capacity=100000, batch_size=20):

        self.model       = model
        self.target_dist = target_dist
        self.init_target_dist = target_dist
        self.horizon     = horizon
        self.replay_buffer = ReplayBuffer(capacity)
        self.batch_size    = batch_size

        self.basis = Basis(self.model.explr_space, num_basis=num_basis)
#         self.lamk  = 1.0/(np.linalg.norm(self.basis.k, axis=1) + 1)**(3.0/2.0)
        self.lamk = np.exp(-0.8*np.linalg.norm(self.basis.k, axis=1))
        self.barr = Barrier(self.model.explr_space)
        # initialize the control sequence
        # self.u_seq = [np.zeros(self.model.action_space.shape)
        #                 for _ in range(horizon)]
        self.u_seq = [0.0*self.model.action_space.sample()
                        for _ in range(horizon)]
        if weights is None:
            weights = {'R' : np.eye(self.model.action_space.shape[0])}
        self.Rinv = np.linalg.inv(weights['R'])
        self.Q = 1.

        self._phik = None
        self.ck = None
        self.init_phik = None

    def reset(self):
        self.u_seq = [0.0*self.model.action_space.sample()
                for _ in range(self.horizon)]
        self.replay_buffer.reset()

    @property
    def phik(self):
        return self._phik

    @phik.setter
    def phik(self, phik):
        assert len(phik) == self.basis.tot_num_basis, 'phik does not have the same number as ck'
        self._phik = phik


    def __call__(self, state, meann, covv, R, Q, sensor_range, ck_list=None, agent_num=None, get_useq=False):
        assert self.phik is not None, 'Forgot to set phik, use set_target_phik method'

        self.u_seq[:-1] = self.u_seq[1:]
        self.u_seq[-1]  = np.zeros(self.model.action_space.shape)

        x = self.model.reset(state)

        mean = meann.copy()
        cov = covv.copy()

        pred_traj = []
        dfk = []
        fdx = []
        fdu = []
        dbar= []
        for t in range(self.horizon):
            # predict observation
            observations = []
            landmarks = mean[3:].reshape(-1,2)
            osbv_lm = []
            idx = -1
            for lm in landmarks:
                idx += 1
                dist = np.sqrt((x[0]-lm[0])**2 + (x[1]-lm[1])**2)_
                if dist <= sensor_range:
                    obsv_lm.append(idx)
            obsv_lm = np.array(obsv_lm)
            # collect all the information that is needed
            pred_traj.append(x[self.model.explr_idx])
            xT = x.copy()
            dfk.append(self.basis.dfk(x[self.model.explr_idx]))
            fdx.append(self.model.fdx(x, self.u_seq[t]))
            fdu.append(self.model.fdu(x))
            dbar.append(self.barr.dx(x[self.model.explr_idx]))
            # step the model forwards
            x = self.model.step(self.u_seq[t] * 1.)
            # ekf prediction
            ctrl = self.u_seq[t]
            G = np.eye(mean.shape[0])
            G[0][2] = -sin(mean[2]) * ctrl[0] * 0.1
            G[1][2] =  cos(mean[2]) * ctrl[0] * 0.1
            num_lm = int((mean.shape[0]-3) / 2)
            BigR = np.block([
                [R, np.zeros((3, 2*num_lm))],
                [np.zeros((2*num_lm, 3)), np.zeros((2*num_lm,2*num_lm))]
            ])
            cov = G @ cov @ G.T + BigR
            mean[0:3] = x.copy()
            mean[2] = normalize_angel(mean[2])
            # ekf correction
            num_obsv = obsv_lm.shape[0]
            H = np.zeros((num_obsv*2, mean.shape[0]))
            idx = -2
            for lid in obsv_lm:
                idx += 2
                lm = mean[3+lid*2 : 5+lid*2]
                zr = np.sqrt((x[0]-lm[0])**2 + (r[1]-lm[1])**2)

                H[idx][0]       = (x[0]-lm[0]) / zr
                H[idx][1]       = (x[1]-lm[1]) / zr
                H[idx][2]       = 0
                H[idx][3+2*lid] = -(x[0]-lm[0]) / zr
                H[idx][4+2*lid] = -(x[1]-lm[1]) / zr

                H[idx+1][0]         = -(x[1]-lm[1]) / zr**2
                H[idx+1][1]         =  (x[0]-lm[0]) / zr**2
                H[idx+1][2]         = -1
                H[idx+1][3+2*lid]   =  (x[1]-lm[1]) / zr**2
                H[idx+1][4+2*lid]   = -(x[0]-lm[0]) / zr**2

            BigQ = block_diag(*[Q for _ in range(num_obsv)])
            K = cov @ H.T @ np.linalg.inv(H @ cov @ H.T + BigQ)
            cov = cov - K @ H @ cov

        # sample any past experiences
        if len(self.replay_buffer) > self.batch_size:
            past_states = self.replay_buffer.sample(self.batch_size)
            pred_traj = pred_traj + past_states
        else:
            past_states = self.replay_buffer.sample(len(self.replay_buffer))
            pred_traj = pred_traj + past_states

        # calculate the cks for the trajectory *** this is also in the utils file
        N = len(pred_traj)
        ck = np.sum([self.basis.fk(xt) for xt in pred_traj], axis=0) / N
        # print('ck: ', len(pred_traj))
        self.ck = ck.copy()
        if ck_list is not None:
            ck_list[agent_num] = ck
            ck = np.mean(ck_list, axis=0)

        fourier_diff = self.lamk * (ck - self.phik)
        fourier_diff = fourier_diff.reshape(-1,1)

        # backwards pass
        rho = np.zeros(self.model.observation_space.shape)
        # xT = pred_traj[-1]
        # print("x(T) = ", pred_traj[-1])
        # print("xd = ", self.xd)
        # rho = self.P1 @ (xT - self.xd)
        for t in reversed(range(self.horizon)):
            edx = np.zeros(self.model.observation_space.shape)
            edx[self.model.explr_idx] = np.sum(dfk[t] * fourier_diff, 0)

            bdx = np.zeros(self.model.observation_space.shape)
            bdx[self.model.explr_idx] = dbar[t]
            rho = rho - self.model.dt * (-self.Q*(edx+bdx) - np.dot(fdx[t].T, rho))

            self.u_seq[t] = -np.dot(np.dot(self.Rinv, fdu[t].T), rho)
            # if (np.abs(self.u_seq[t]) > 1.0).any():
            if (np.abs(self.u_seq[t]) > 2.0).any():
                self.u_seq[t] /= np.linalg.norm(self.u_seq[t])

        self.replay_buffer.push(state[self.model.explr_idx])

        if get_useq is True:
            return self.u_seq[0].copy(), self.u_seq.copy()
        else:
            return self.u_seq[0].copy()
