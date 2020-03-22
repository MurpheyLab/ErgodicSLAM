import numpy as np
from numpy import sin, cos
from gym.spaces import Box

class DoubleIntegratorSE2(object):
    def __init__(self, size=1.0):
        self.observation_space = Box(np.array([0., 0., 0., -np.inf, -np.inf, -np.inf]),
                                     np.array([size, size, size, np.inf, np.inf, np.inf]),
                                     dtype=np.float32 )

        self.action_space = Box(np.array([-size, -size]),
                                np.array([size, size]),
                                dtype=np.float32 )

        self.explr_space = Box(np.array([0., 0.]),
                               np.array([size, size]),
                               dtype=np.float32)

        self.explr_idx  = [0, 1]

        self.dt = 0.1

        self.reset()

    def fdx(self, x, u):
        '''
        State linearization
        '''
        return np.array([
            [ 0., 0.,              0.,0.8, 0., 0.],
            [ 0., 0.,              0., 0.,0.8, 0.],
            [ 0., 0.,              0., 0., 0.,0.8],
            [ 0., 0., -sin(x[2])*u[0], 0., 0., 0.],
            [ 0., 0.,  cos(x[2])*u[0], 0., 0., 0.],
            [ 0., 0.,              0., 0., 0., 0.]
        ])

    def fdu(self, x):
        '''
        Control linearization
        '''
        self.B = np.array([
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [cos(x[2]), 0.],
            [sin(x[2]), 0.],
            [0., 1.]
        ])
        return self.B.copy()

    def reset(self, state=None):
        '''
        Resets the property self.state
        '''
        if state is None:
            self.state = np.zeros(self.observation_space.shape[0])
            self.state[:3] = np.random.uniform(0., 0.9, size=(3,))
        else:
            self.state = state.copy()

        return self.state.copy()

    def f(self, x, u):
        '''
        Continuous time dynamics
        '''
        return np.array([0.8*x[3], 0.8*x[4], 0.8*x[5], cos(x[2])*u[0], sin(x[2])*u[0], u[1]])

    def step(self, a):
        '''
        Basic euler step
        '''
        # TODO: include ctrl clip
        # print("self.f: ", self.f(self.state, a))
        self.state = self.state + self.f(self.state, a) * self.dt
        # self.state = self.state + np.array([cos(self.state[2])*a[0], sin(self.state[2])*a[0], a[1], 0, 0, 0]) * self.dt
        return self.state.copy()

    def noisy_step(self, a, noise):
        '''
        Basic euler step with noise
        '''
        self.state = self.state + self.f(self.state, a) * self.dt + noise * np.random.randn(len(self.state))
        return self.state.copy()