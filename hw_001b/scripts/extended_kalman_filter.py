#!/usr/bin/env python
import rospy

from sensor_msgs.msg import Imu
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from numpy import linalg as la
# from scipy.stats import norm
# from sympy import Symbol, symbols, Matrix, sin, cos
# from sympy.interactive import printing
# printing.init_printing()

# phi_dot, theta_dot, psi_dot, phi, theta, psi, dt = symbols('\dot\phi \dot\\theta \dot\psi \phi \\theta \psi \delta')

# gs = Matrix([[phi_dot],
#              [theta_dot],
#              [psi_dot],
#              [phi + dt * phi_dot],
#              [theta + dt * theta_dot],
#              [psi + dt * psi_dot]])
# state = Matrix([phi_dot, theta_dot, psi_dot, phi, theta, psi])

# print np.matrix(gs.jacobian(state))

# print ""

# hs = Matrix([[phi_dot],
#              [theta_dot],
#              [psi_dot]])
# print np.matrix(hs.jacobian(state))


class EKF:
    def __init__(self):
        self.sub = rospy.Subscriber("imu0", Imu, self.ReceiveImuData)
        self.is_first = True
        self.x_t = [0 for i in xrange(6)]
        self.time = 0
        self.previous_time = -1

        self.R = np.diag([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]) # Process Covariance
        self.Q = np.diag([0.01, 0.01, 0.01]) # Measurement Covariance
        self.P_cov = np.eye(6) * 0.000001
        self.original = []
        self.estiamtes = []
        self.error_cov = []
        self.num_iter = 6000
        self.iter_count = 0
        self.dt = 0
    def ReceiveImuData(self, msg):
        # Perform Prediction Step
        self.time = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
        if self.is_first:
            self.x_t[0] = msg.angular_velocity.x
            self.x_t[1] = msg.angular_velocity.y
            self.x_t[2] = msg.angular_velocity.z
            self.x_t[3] = 0
            self.x_t[4] = 0
            self.x_t[5] = 0
            self.is_first = False
        else:

            # Compute dt
            dt = self.time - self.previous_time
            self.dt = dt
            # Compute Predicted State using g(x)
            self.x_t[0] = self.x_t[0]
            self.x_t[1] = self.x_t[1]
            self.x_t[2] = self.x_t[2]
            self.x_t[3] = self.x_t[3] + dt * self.x_t[0]
            self.x_t[4] = self.x_t[4] + dt * self.x_t[1]
            self.x_t[5] = self.x_t[5] + dt * self.x_t[2]

            # Compute Jacobian
            Jg = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [dt , 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, dt , 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, dt , 0.0, 0.0, 1.0]])

            # Predicted error covariance
            self.P_cov = Jg * self.P_cov * Jg.T + self.R

            # Compute H jacobian
            Jh = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
            # Comppute kalman Gain
            K_t = self.P_cov * Jh.T * la.inv(Jh * self.P_cov * Jh.T + self.Q)

            h = np.matrix([self.x_t[0], self.x_t[1], self.x_t[2]]).T
            # Compute Corrected State estimate
            z_t = np.matrix([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]).T
            x_t = np.matrix(self.x_t).T + K_t * (z_t - h)
            self.P_cov = (np.eye(6) - K_t * Jh) * self.P_cov

            self.x_t = x_t.T.tolist()[0]

            # Store results
            self.original.append(z_t.T.tolist()[0])
            self.estiamtes.append(self.x_t)
            self.error_cov.append(self.P_cov)

        self.previous_time = self.time
        self.iter_count += 1
        if self.iter_count == self.num_iter:
            self.SaveData()
    def SaveData(self):
        x_axis = np.arange(0, len(self.original)*self.dt, self.dt)

        plt.figure()
        original, = plt.plot(x_axis, [x[0] for x in self.original])
        estimate, = plt.plot(x_axis, [x[0] for x in self.estiamtes])
        integral, = plt.plot(x_axis, [x[3] for x in self.estiamtes])
        plt.xlabel(r'Iterations$(s)$')
        plt.ylabel(r'angle$^{rad}$')
        plt.title("Roll - EKF Estimate")
        plt.legend([original, estimate, integral], [r'IMU - $\dot{\phi}$', r'Estimated - $\dot{\phi}$', r'Estimated - $\phi$'])
        plt.savefig('roll-result.png', dpi=500)

        plt.figure()
        original, = plt.plot(x_axis, [x[1] for x in self.original])
        estimate, = plt.plot(x_axis, [x[1] for x in self.estiamtes])
        integral, = plt.plot(x_axis, [x[4] for x in self.estiamtes])
        plt.xlabel(r'Iterations$(s)$')
        plt.ylabel(r'angle$^{rad}$')
        plt.title("Pitch - EKF Estimate")
        plt.legend([original, estimate, integral], [r'IMU - $\dot{\theta}$', r'Estimated - $\dot{\theta}$', r'Estimated - $\theta$'])
        plt.savefig('pitch-result.png', dpi=500)

        plt.figure()
        original, = plt.plot(x_axis, [x[2] for x in self.original])
        estimate, = plt.plot(x_axis, [x[2] for x in self.estiamtes])
        integral, = plt.plot(x_axis, [x[5] for x in self.estiamtes])
        plt.xlabel(r'Iterations$(s)$')
        plt.ylabel(r'angle$^{rad}$')
        plt.title("Yaw - EKF Estimate")
        plt.legend([original, estimate, integral], [r'IMU - $\dot{\psi}$', r'Estimated - $\dot{\psi}$', r'Estimated - $\psi$'])
        plt.savefig('yaw-result.png', dpi=500)

def main():
    print "Extended Kalman Filter"
    rospy.init_node('ekf')

    EKF()
    rospy.spin()

if __name__ == '__main__':
    main()