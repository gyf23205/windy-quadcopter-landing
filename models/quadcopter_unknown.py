import casadi
import numpy as np
from quadcopter import QuadCopter

class QuadCopterWindUnknown(QuadCopter):
    def __init__(self, init_state, Q0, R0, Q1, R1, N, v_max0, r_max0, v_max1, r_max1, init_wind_param):
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
        self.wind_hist = None# 2D wind velocity
        self.wind_params = init_wind_param # np.array([[8, 2, 0.7], [0.8, 0.3, 0.2]])
        self.threshold = 2
        self.hybrid = True
        self.setup_wind()


    def est_vw(self, states_pre, states_now, u):
        tau = self.L * (u[1] - u[0])
        d_x = states_pre[3]
        d_z = states_pre[4]
        d_theta = states_pre[5]
        d_vx = (u[0] + u[1]) * np.sin(states_pre[2])/self.m
        d_vz = ((u[0] + u[1]) * np.cos(states_pre[2])- self.m * self.g)/self.m
        d_vtheta = tau/self.I
        d_states = np.concatenate([d_x, d_z, d_theta, d_vx, d_vz, d_vtheta], axis=-1)
        states_pred = states_pre + d_states * self.dt
        res = states_now - states_pred
        F_w = res[3:5] * self.m
        
        A_e = np.array([np.abs(self.A * np.cos(self.states[2])), np.abs(self.A * np.sin(self.states[2]))])
        v_w = -(2 * F_w)/(self.rho * self.C_d * A_e)
        v_w = np.sign(v_w) * np.sqrt(np.abs(v_w))

        if self.wind_hist is None:
            self.wind_hist = v_w
        else:
            self.wind_hist = np.concatenate(self.wind_hist, v_w)
        return

    def est_wind_params(self):
        for i in range(self.wind_hist.shape[0]):
            mu = casadi.MX.sym('mu')
            sigma = casadi.MX.sym('sigma')
            tau_c = casadi.MX.sym('tau_c')
            params = casadi.vertcat(mu, sigma, tau_c)

            # Define symbolic expressions for log-likelihood components
            alpha = casadi.exp(-self.dt / tau_c)  # Autoregressive coefficient
            sigma_epsilon2 = sigma**2 * (1 - alpha**2)  # Noise variance

            # Ensure valid parameter ranges
            eps = 1e-6  # Small constant to prevent numerical issues
            # valid_sigma = casadi.if_else(sigma > eps, sigma, eps)
            # valid_tau_c = casadi.if_else(tau_c > eps, tau_c, eps)
            sigma_epsilon2 = casadi.if_else(sigma_epsilon2 > eps, sigma_epsilon2, eps)

            # Residuals for the log-likelihood
            residuals = self.wind_hist[i, 1:] - alpha * self.wind_hist[0, :-1] - (1 - alpha) * mu
            log_likelihood = -0.5 * (len(self.wind_hist)-1) * casadi.log(2 * np.pi * sigma_epsilon2) - 0.5 * casadi.sum1(residuals**2 / sigma_epsilon2)

            # Define optimization problem
            objective = -log_likelihood  # Negative log-likelihood for minimization
            nlp = {'x': params, 'f': objective}

            # Create CasADi solver
            solver = casadi.nlpsol('solver', 'ipopt', nlp)

            # Initial guesses and bounds
            initial_guess = self.wind_params[0, :]  # Initial guesses for [mu, sigma, tau_c]
            lbx = [-np.inf, eps, eps]  # Lower bounds for [mu, sigma, tau_c]
            ubx = [np.inf, np.inf, np.inf]  # Upper bounds

            # Solve the optimization problem
            solution = solver(x0=initial_guess, lbx=lbx, ubx=ubx)
            self.wind_params[i, :] = solution['x'].full().flatten()

    def update_param(self, x0, ref, k, N):
        p = casadi.vertcat(x0)
        for l in range(N):
            if k+l < ref.shape[0]:
                ref_state = ref[k+l, :]
            else:
                ref_state = ref[-1, :]
            xt = casadi.DM(ref_state[0:6])
            p = casadi.vertcat(p, xt)
        
        # Q and R matrix
        if self.mode==0:
            p = casadi.vertcat(p, self.Q0, self.R0, self.R0)
        else:
            p = casadi.vertcat(p, self.Q1, self.R1, self.R1)

        # Estimated wind of current time
        if k==0: # Cannot estimate wind at first, directly sample from the initial guess
            v_w = np.random.normal(self.wind_params[0, :], self.wind_params[1, :])
        else:
            
        return p
    

    def setup_wind(self):
        # Define the dynamics

        # states = [x, z, theta, x_dot, z_dot, theta_dot]
        v_w = casadi.SX.sym('vw', 2)
        states = casadi.SX.sym('x', 6) # 3D position (x, y, z), row (theta), pitch (phi), yaw (psi)
        u = casadi.SX.sym('u', 2) # Forces
        self.n_states = states.numel()
        self.n_controls = u.numel()

        # Compute the effect of wind
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

        self.f = casadi.Function('f', [states, u, v_w], [d_states])

        # Define the optimal control problem
        X = casadi.SX.sym('X', self.n_states, self.N + 1)
        U = casadi.SX.sym('U', self.n_controls, self.N)
        P = casadi.SX.sym('P', (self.N + 1) * self.n_states + self.n_states + self.n_controls) # Reference path of current horizen + diag of Q + R + current estimated wind
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

            k_1 = self.f(state, control, P[-2, len(P)])
            k_2 = self.f(state + self.dt/2 * k_1, control, P[-2, len(P)])
            k_3 = self.f(state + self.dt/2 * k_2, control, P[-2, len(P)])
            k_4 = self.f(state + self.dt * k_3, control, P[-2, len(P)])
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