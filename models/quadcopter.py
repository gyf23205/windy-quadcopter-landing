import casadi
import numpy as np
from plan_dubin import plan_dubins_path

class QuadCopter(object):
    def __init__(self, init_state, Q, R, N, v_max, r_max):
        '''
        Q: np array with size (3,), containing the diagnal elements of the PSD matrix which is used to penalize the states in the quadratic MPC cost.
        R: Scalar, penalize inputs in the MPC cost.
        N: Time horizen of the MPC solver.
        '''
        self.states = init_state
        self.m = 1.5 # Mass in kg
        self.L = 0.4 # Arms length in meter
        self.dt = 0.1
        self.g = 9.81
        self.I = 0.03
        self.rho = 1.225
        self.A = 0.1
        self.C_d = 1.3
        self.C_r = 0.03
        self.F_max = 10
        self.W_max = 10
        self.v_max = v_max
        self.r_max = r_max

        self.Q = Q
        self.R = R
        self.N = N
        self.setup()

    
    def update_param(self, x0, ref, k, N):
        p = casadi.vertcat(x0)
        for l in range(N):
            if k+l < ref.shape[0]:
                ref_state = ref[k+l, :]
            else:
                ref_state = ref[-1, :]
            xt = casadi.DM(ref_state[0:6])
            p = casadi.vertcat(p, xt)
        return p
    

    def shift_timestep(self, time, state, control):
        delta_state = self.f(state, control[:, 0])
        next_state = casadi.DM.full(state + self.dt * delta_state)
        next_time = time + self.dt
        next_control = casadi.horzcat(control[:, 1:],
                                    casadi.reshape(control[:, -1], -1, 1))
        return next_time, next_state, next_control
    

    def setup(self):
        # Define the dynamics

        # states = [x, z, theta, x_dot, z_dot, theta_dot]
        states = casadi.SX.sym('x', 6) # 3D position (x, y, z), row (theta), pitch (phi), yaw (psi)
        u = casadi.SX.sym('u', 2) # Forces
        self.n_states = states.numel()
        self.n_controls = u.numel()

        tau = self.L * (u[1] - u[0])
        d_x = states[3]
        d_z = states[4]
        d_theta = states[5]
        d_vx = (u[0] + u[1]) * casadi.sin(states[2])/self.m
        d_vz = ((u[0] + u[1]) * casadi.cos(states[2])- self.m * self.g)/self.m
        d_vtheta = tau/self.I
        d_states = casadi.vertcat(d_x, d_z, d_theta, d_vx, d_vz, d_vtheta)

        self.f = casadi.Function('f', [states, u], [d_states])

        # Define the optimal control problem
        X = casadi.SX.sym('X', self.n_states, self.N + 1)
        U = casadi.SX.sym('U', self.n_controls, self.N)
        P = casadi.SX.sym('P', (self.N + 1) * self.n_states) # Reference path of current horizen
        Q = casadi.diagcat(*self.Q)
        R = casadi.diagcat(self.R, self.R)

        cost = 0
        g = X[:, 0] - P[:self.n_states]

        for k in range(self.N):
            state = X[:, k]
            control = U[:, k]
            cost = cost + (state - P[(k+1)*self.n_states:(k+2)*self.n_states]).T @ Q @ (state - P[(k+1)*self.n_states:(k+2)*self.n_states]) + \
                    control.T @ R @ control
            next_state = X[:, k + 1]

            k_1 = self.f(state, control)
            k_2 = self.f(state + self.dt/2 * k_1, control)
            k_3 = self.f(state + self.dt/2 * k_2, control)
            k_4 = self.f(state + self.dt * k_3, control)
            predicted_state = state + self.dt/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            g = casadi.vertcat(g, next_state - predicted_state)

        opt_variables = casadi.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))

        nlp_prob = {
            'f': cost,
            'x': opt_variables,
            'g': g,
            'p': P
        }

        opts = {
            'ipopt': {
                'sb': 'yes',
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0,
        }
        self.solver = casadi.nlpsol('solver', 'ipopt', nlp_prob, opts)

    
    def solve(self, X0, u0, ref, idx):
        lbx = casadi.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))
        ubx = casadi.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))

        for i in range(self.n_states):
            if i in [0, 1, 2]:
                lbx[i:self.n_states * (self.N + 1):self.n_states] = -casadi.inf
                ubx[i:self.n_states * (self.N + 1):self.n_states] = casadi.inf
            elif i in [3, 4]:
                lbx[i:self.n_states * (self.N + 1):self.n_states] = -self.v_max
                ubx[i:self.n_states * (self.N + 1):self.n_states] = self.v_max
            else:
                lbx[i:self.n_states * (self.N + 1):self.n_states] = -self.r_max
                ubx[i:self.n_states * (self.N + 1):self.n_states] = self.r_max

        lbx[self.n_states * (self.N + 1):self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = 0
        ubx[self.n_states * (self.N + 1):self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = self.F_max
        lbx[self.n_states * (self.N + 1) + 1:self.n_states * (self.N + 1) + self.n_controls *self.N:self.n_controls] = 0
        ubx[self.n_states * (self.N + 1) + 1:self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = self.F_max

        lbg = casadi.DM.zeros((self.n_states * (self.N+1)))
        ubg = -casadi.DM.zeros((self.n_states * (self.N+1)))

        args = {
            'lbg': lbg,
            'ubg': ubg,
            'lbx': lbx,
            'ubx': ubx
        }

        args['p'] = self.update_param(X0[:,0], ref, idx, self.N)
        args['x0'] = casadi.vertcat(casadi.reshape(X0, self.n_states * (self.N + 1), 1),
                                        casadi.reshape(u0, self.n_controls * self.N, 1))

        sol = self.solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                        lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

        u = casadi.reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N)
        X = casadi.reshape(sol['x'][:self.n_states * (self.N + 1)], self.n_states, self.N + 1)
        return u, X 


class QuadCopterWind(QuadCopter):
    def __init__(self, init_state, Q0, R0, Q1, R1, N, v_max0, r_max0, v_max1, r_max1, wind):
        super().__init__(init_state, Q0, R0, N, v_max0, r_max0)
        self.v_max0 = v_max0
        self.r_max0 = r_max0
        self.v_max1 = v_max1
        self.r_max1 = r_max1
        self.Q0 = casadi.DM(Q0)
        self.R0 = casadi.DM(R0)
        self.Q1 = casadi.DM(Q1)
        self.R1 = casadi.DM(R1)
        self.mode = 0
        self.wind = wind # 2D wind velocity
        self.threshold = 2
        self.hybrid = True
        self.setup_wind()


    def update_param(self, x0, ref, k, N):
        p = casadi.vertcat(x0)
        for l in range(N):
            if k+l < ref.shape[0]:
                ref_state = ref[k+l, :]
            else:
                ref_state = ref[-1, :]
            xt = casadi.DM(ref_state[0:6])
            p = casadi.vertcat(p, xt)
        if self.mode==0:
            p = casadi.vertcat(p, self.Q0, self.R0, self.R0)
        else:
            p = casadi.vertcat(p, self.Q1, self.R1, self.R1)
        return p
    

    def setup_wind(self):
        # Define the dynamics

        # states = [x, z, theta, x_dot, z_dot, theta_dot]
        v_w = casadi.DM(self.wind)
        states = casadi.SX.sym('x', 6) # 3D position (x, y, z), row (theta), pitch (phi), yaw (psi)
        u = casadi.SX.sym('u', 2) # Forces
        self.n_states = states.numel()
        self.n_controls = u.numel()

        # Compute the effect of wind
        phi = -casadi.atan2(v_w[1], v_w[0])
        A_ex = casadi.fabs(self.A * casadi.cos(self.states[2]))
        A_ez = casadi.fabs(self.A * casadi.sin(self.states[2]))
        v_rel = states[0:2] - v_w
        F_x = -0.5 * self.rho * self.C_d * A_ex * v_rel[0] * casadi.fabs(v_rel[0])
        F_z = -0.5 * self.rho * self.C_d * A_ez * v_rel[1] * casadi.fabs(v_rel[1])
        
        tau = self.L * (u[1] - u[0])
        d_x = states[3]
        d_z = states[4]
        d_theta = states[5]
        d_vx = ((u[0] + u[1]) * casadi.sin(states[2]) + F_x)/self.m
        d_vz = ((u[0] + u[1]) * casadi.cos(states[2])- self.m * self.g + F_z)/self.m
        d_vtheta = tau/self.I
        d_states = casadi.vertcat(d_x, d_z, d_theta, d_vx, d_vz, d_vtheta)

        self.f = casadi.Function('f', [states, u], [d_states])

        # Define the optimal control problem
        X = casadi.SX.sym('X', self.n_states, self.N + 1)
        U = casadi.SX.sym('U', self.n_controls, self.N)
        P = casadi.SX.sym('P', (self.N + 1) * self.n_states + self.n_states + self.n_controls) # Reference path of current horizen + diag of Q + R
        Q = casadi.diag(P[-(self.n_controls+self.n_states):-self.n_controls])
        R = casadi.diagcat(P[-1], P[-1])

        cost = 0
        g = X[:, 0] - P[:self.n_states]

        for k in range(self.N):
            state = X[:, k]
            control = U[:, k]
            cost = cost + (state - P[(k+1)*self.n_states:(k+2)*self.n_states]).T @ Q @ (state - P[(k+1)*self.n_states:(k+2)*self.n_states]) + \
                    control.T @ R @ control
            next_state = X[:, k + 1]

            k_1 = self.f(state, control)
            k_2 = self.f(state + self.dt/2 * k_1, control)
            k_3 = self.f(state + self.dt/2 * k_2, control)
            k_4 = self.f(state + self.dt * k_3, control)
            predicted_state = state + self.dt/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            g = casadi.vertcat(g, next_state - predicted_state)

        opt_variables = casadi.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))

        nlp_prob = {
            'f': cost,
            'x': opt_variables,
            'g': g,
            'p': P
        }

        opts = {
            'ipopt': {
                'sb': 'yes',
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0,
        }
        self.solver = casadi.nlpsol('solver', 'ipopt', nlp_prob, opts)

            
    def solve(self, X0, u0, ref, idx):
        lbx = casadi.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))
        ubx = casadi.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))
        # Determine the mode of current iteration based on the difference between current states and the reference
        if self.hybrid:
            if np.linalg.norm(X0[:, 0] - ref[idx, :]) < self.threshold or len(ref) - idx < 10:
                self.mode = 0
            else:
                self.mode = 1
        else:
            self.mode = 0
        self.mode = 0
        # print(self.mode)
        if self.mode==0:
            for i in range(self.n_states):
                if i in [0, 1, 2]:
                    lbx[i:self.n_states * (self.N + 1):self.n_states] = -casadi.inf
                    ubx[i:self.n_states * (self.N + 1):self.n_states] = casadi.inf
                elif i in [3, 4]:
                    lbx[i:self.n_states * (self.N + 1):self.n_states] = -self.v_max0
                    ubx[i:self.n_states * (self.N + 1):self.n_states] = self.v_max0
                else:
                    lbx[i:self.n_states * (self.N + 1):self.n_states] = -self.r_max0
                    ubx[i:self.n_states * (self.N + 1):self.n_states] = self.r_max0
        else:
            for i in range(self.n_states):
                if i in [0, 1, 2]:
                    lbx[i:self.n_states * (self.N + 1):self.n_states] = -casadi.inf
                    ubx[i:self.n_states * (self.N + 1):self.n_states] = casadi.inf
                elif i in [3, 4]:
                    lbx[i:self.n_states * (self.N + 1):self.n_states] = -self.v_max1
                    ubx[i:self.n_states * (self.N + 1):self.n_states] = self.v_max1
                else:
                    lbx[i:self.n_states * (self.N + 1):self.n_states] = -self.r_max1
                    ubx[i:self.n_states * (self.N + 1):self.n_states] = self.r_max1

        lbx[self.n_states * (self.N + 1):self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = 0
        ubx[self.n_states * (self.N + 1):self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = self.F_max
        lbx[self.n_states * (self.N + 1) + 1:self.n_states * (self.N + 1) + self.n_controls *self.N:self.n_controls] = 0
        ubx[self.n_states * (self.N + 1) + 1:self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = self.F_max

        lbg = casadi.DM.zeros((self.n_states * (self.N+1)))
        ubg = -casadi.DM.zeros((self.n_states * (self.N+1)))

        args = {
            'lbg': lbg,
            'ubg': ubg,
            'lbx': lbx,
            'ubx': ubx
        }

        args['p'] = self.update_param(X0[:,0], ref, idx, self.N)
        args['x0'] = casadi.vertcat(casadi.reshape(X0, self.n_states * (self.N + 1), 1),
                                        casadi.reshape(u0, self.n_controls * self.N, 1))

        sol = self.solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                        lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

        u = casadi.reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N)
        X = casadi.reshape(sol['x'][:self.n_states * (self.N + 1)], self.n_states, self.N + 1)
        return u, X 

if __name__=='__main__':
    print("Dubins path planner sample start!!")
    import matplotlib.pyplot as plt

    start_x = 1.0  # [m]
    start_y = 1.0  # [m]
    start_yaw = np.deg2rad(45.0)  # [rad]

    end_x = -3.0  # [m]
    end_y = -3.0  # [m]
    end_yaw = np.deg2rad(-45.0)  # [rad]

    curvature = 1.0

    path_x, path_y, path_yaw, mode, lengths = plan_dubins_path(start_x, start_y, start_yaw,
                                                               end_x, end_y, end_yaw, curvature)

