
import numpy as np
# import pandas as pd
from scipy import integrate


class LinearSystemSimulator:
    def __init__(self, A, B, C=None, D=None):
        self.A = np.array(A)  # system matrix
        self.B = np.array(B)  # input matrix

        if C is None:
            C = np.eye(len(A))

        if D is None:
            D = np.zeros_like(B)

        self.C = np.atleast_2d(np.array(C))  # measurement matrix
        self.D = np.atleast_2d(np.array(D))  # feedforward matrix

        self.p, self.n = C.shape  # [# of outputs, # of states]
        self.m = B.shape[0]  # of inputs

        # To store simulation data
        self.N = []
        self.t = []
        self.x = []
        self.y = []
        self.u = []
        self.state = []

    def system_ode(self, x, t, tsim, usim):
        '''
        Dynamical system model of a linear system.

        Inputs
            x: states (array)
            t: time (array)
            tsim: numpy array of simulation time
            usim: numpy array with each input in a column

        Outputs
            xdot: derivative of states (array)

        '''

        # Find current inputs
        time_index = np.argmin(np.abs(t - tsim))  # find closest time to simulation
        usim = np.atleast_2d(usim)
        u = usim[time_index, :]
        u = np.atleast_2d(u).T

        # Derivative of states
        x = np.atleast_2d(x)
        x = np.transpose(x)
        xdot = (self.A @ x) + (self.B @ u)

        # Current output
        # y = (self.C @ x) + (self.D @ u)

        # Return the state derivative
        return np.squeeze(xdot)

    def measurement_function(self, x, u, measurement_type='linear'):
        # Set the measurement function & compute it
        if measurement_type == 'linear':  # use given C & D matrices to compute output
            y = (self.C @ x) + (self.D @ u)
        elif measurement_type == 'divide_first_two_states':
            self.y = np.atleast_2d(self.y[:, 0]).T  # only need 1st colum
            y = x[1] / x[0]
        elif measurement_type == 'multiply_first_two_states':
            self.y = np.atleast_2d(self.y[:, 0]).T  # only need 1st column
            y = x[1] * x[0]

        return y

    def simulate(self, x0, tsim, usim, measurement_type='linear'):
        self.N = len(tsim)
        self.t = np.atleast_2d(tsim).T  # simulation time vector
        self.u = np.atleast_2d(usim)  # input(s)
        self.y = np.zeros((self.N, self.p))  # preallocate output

        # Solve ODE
        self.x = integrate.odeint(self.system_ode, x0, tsim, args=(tsim, usim))

        # Compute output
        for k in range(self.N):
            # self.y[k, :] = (self.C @ self.x[k, :]) + (self.D @ self.u[k, :])
            self.y[k, :] = self.measurement_function(self.x[k, :], self.u[k, :].T, measurement_type)

        # Store state
        self.state = {'t': tsim,
                      'u': self.u,
                      'x0': x0,
                      'x': self.x,
                      'y': self.y}

        return self.state, self.y
