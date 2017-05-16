import numpy as np
from numpy.linalg import inv

class KalmanFilter2D(object):

    def __init__(self, measurements_interval, starting_position):
        # estimates for x, y, x_velocity, y_velocity
        self.x = np.array([[starting_position[0]], [starting_position[1]], [0.], [0.]])
        self.dt = measurements_interval

        # state transition matrix, x_next = x + x_velocity * dt, y_next = y + y_velocity * dt
        self.F = np.array([[1., 0, dt, 0], [0., 1., 0., dt], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        # measurement function: reflect the fact that we observe x and y but not the two velocities
        self.H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        # initial uncertainty / variances: 0 (very certain) for positions x and y,
        # 1000 (very uncertain) for the two velocities
        self.P = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 1000., 0.], [0., 0., 0., 1000.]])
        # noise matrix 2x2 for measurements
        self.R = np.array([[0.1, 0.], [0., 0.1]])

    def update(self, measurement, motion):
        # prediction update
        self.x = self.F @ self.x + motion

        # covariance matrix update (basically means and variances of current prediction & motion should be summed up)
        self.P = (self.F @ self.P) @ self.F.T

        # measurement update
        # measurement to column vector
        Z = np.array(measurement).reshape(len(measurement), 1)
        # error
        y = Z - (self.H @ self.x)

        # select only covariances for x and y, as when measuring we dont observe velocities
        S = self.H @ self.P @ self.H.T + self.R
        # kalman gain
        K = self.P @ self.H.T @ inv(S)

        # updating our beliefs
        # means
        self.x = self.x + (K @ y)
        # variances
        self.P = (np.identity(4) - (K @ self.H)) @ self.P

if __name__ == "__main__":
    tests = [dict(measurements=[[5., 10.], [6., 8.], [7., 6.], [8., 4.], [9., 2.], [10., 0.]],
                  starting_position=(4., 12.)),
             dict(measurements=[[1., 4.], [6., 0.], [11., -4.], [16., -8.]], starting_position=[-4., 8.]),
             dict(measurements=[[1., 17.], [1., 15.], [1., 13.], [1., 11.]], starting_position=[1., 19.]),]

    dt = 0.1
    motion = np.array([[0.], [0.], [0.], [0.]])

    for test in tests:
        filter = KalmanFilter2D(dt, test['starting_position'])
        for m in test['measurements']:
            filter.update(m, motion)

        print("x =")
        print(filter.x)
        print("P =")
        print(filter.P)


