import numpy as np
import casadi
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib import animation
from r_star import RStarPlanner
from models.quadcopter_unknown import QuadCopterWindUnknown
from matplotlib.lines import Line2D
from plan_dubin import plan_dubins_path
from wind_generator import WindGen



def simulate(ref_states, cat_states, num_frames, reference, save=False):
    def create_triangle(state=[0,0,0], h=2, w=0.5, update=False):
        x, y, th = state
        th += np.pi/2
        triangle = np.array([
            [h, 0   ],
            [0,  w/2],
            [0, -w/2],
            [h, 0   ]
        ]).T
        rotation_matrix = np.array([
            [np.cos(th), -np.sin(th)],
            [np.sin(th),  np.cos(th)]
        ])

        coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T
        if update == True:
            return coords
        else:
            return coords[:3, :]

    def init():
        # hm.set_data(np.ones(world.heatmap.shape))
        return path, horizon#, current_state, target_state,

    def animate(i):
        # get variables
        x = cat_states[0, 0, i]
        y = cat_states[1, 0, i]
        th = cat_states[2, 0, i]

        # get ref variables
        x_ref = ref_states[:, 0]
        y_ref = ref_states[:, 1]


        # update ref path
        ref_path.set_data(x_ref, y_ref)

        # update path
        if i == 0:
            path.set_data(np.array([]), np.array([]))
        x_new = np.hstack((path.get_xdata(), x))
        y_new = np.hstack((path.get_ydata(), y))
        path.set_data(x_new, y_new)

        # update horizon
        x_new = cat_states[0, :, i]
        y_new = cat_states[1, :, i]
        horizon.set_data(x_new, y_new)

        # update current_state
        current_state.set_xy(create_triangle([x, y, th], update=True))

        # update heatmap
        # img = plot_heatmap(world, None, i)
        # hm.set_data(img)
        # # update target_state
        # xy = target_state.get_xy()
        # target_state.set_xy(xy)

        return path, horizon#, current_state, target_state,

    # create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(left = -15, right = 15)
    ax.set_ylim(bottom = -0, top = 15)

    path, = ax.plot([], [], 'r', linewidth=2)

    ref_path, = ax.plot([], [], 'b', linewidth=2)
    #   horizon
    horizon, = ax.plot([], [], 'x-g', alpha=0.5)


    # hm = plt.imshow(np.ones(heatmaps[0].shape)*255, origin='lower', extent=[0., world.len_grid * size_world[0], 0, world.len_grid * size_world[1]])

    #   current_state
    current_triangle = create_triangle(reference[:3])
    current_state = ax.fill(current_triangle[:, 0], current_triangle[:, 1], color='r')
    current_state = current_state[0]
    # #   target_state
    # target_triangle = create_triangle(reference[3:])
    # target_state = ax.fill(target_triangle[:, 0], target_triangle[:, 1], color='b')
    # target_state = target_state[0]
    legend_elements = [Line2D([0], [0], marker='>', color='y', markerfacecolor='y', markersize=15, label='Robots'),
                   Line2D([0], [0], marker='x',color='g', markerfacecolor='g', markersize=15,label='MPC Predicted Path',),
                   ]

    ax.legend(handles=legend_elements, loc='upper right')
    sim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=num_frames,
        #interval=step_horizon*100,
        interval=100,
        blit=False,
        repeat=False
    )

    if save == True:
        sim.save('results/along_wind.gif', fps=30)
    plt.show()
    return sim

def plot_ref(path, thetas, start, goal, x_range, y_range):
        # Plot the result
    plt.figure(figsize=(8, 8))
    # for ox, oy, radius in obstacles:
    #     circle = plt.Circle((ox, oy), radius, color='r', alpha=0.5)
    #     plt.gca().add_patch(circle)
    plt.plot([p[0] for p in path], [p[1] for p in path], 'b-', label="Path")
    plt.scatter([p[0] for p in path], [p[1] for p in path], c='b')
    plt.quiver(
        [p[0] for p in path], [p[1] for p in path],
        np.cos(thetas), np.sin(thetas), angles='xy', scale_units='xy', scale=1, color='g', label='Theta'
    )
    plt.scatter([start[0], goal[0]], [start[1], goal[1]], c='g', label='Start/Goal')
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.legend()
    plt.grid()
    plt.title("R* Path with Orientation")
    plt.show()

def main(args=None):
    dt = 0.1
    N = 8
    t0 = 0
    wind_gen = WindGen(np.array([[10, 2, 0.5], [1, 0.3, 0.5]]), 0.1)

    # x in [0, size_world[0]], y in [0, size_world[1] * world.len_grid]
    x_0 = -10
    y_0 = 10
    theta_0 = 0

    # x in [0, size_world[0]], y in [0, size_world[1]* world.len_grid]
    x_goal = 0
    y_goal = 0
    theta_goal = 0

    r_max0 = 1 # Max angular velocity
    v_max0 = 1 # Max linear velocity
    r_max1 = 2 # Max angular velocity
    v_max1 = 2 # Max linear velocity
    
    path_x, path_y, path_yaw, _, _ = plan_dubins_path(x_0, y_0, theta_0, x_goal, y_goal, theta_goal, r_max0, step_size=v_max0*dt)
    ref_states = np.array([path_x, path_y, -path_yaw]).T
    ref_states = np.concatenate([ref_states, np.zeros(ref_states.shape)], axis=-1)
    # plot_ref(ref_states.T[:, 0:2], ref_states.T[:, 2], start, goal, x_range, y_range)

    Q0 = [7, 7, 7, 0, 0, 0] # No penalty on velocities
    R0 = 2
    Q1 = [10, 10, 10, 0, 0, 0] # No penalty on velocities
    R1 = 0.5


    # # TODO Move the definition of obstacles to env, not in agents
    init_state = [x_0, y_0, theta_0, 0, 0, 0]
    qc = QuadCopterWindUnknown(init_state, Q0, R0, Q1, R1, N, v_max0, r_max0, v_max1, r_max1)

    state_0 = casadi.DM(init_state)
    u0 = casadi.DM.zeros((qc.n_controls, N))
    X0 = casadi.repmat(state_0, 1, N + 1)
    cat_states = np.array(X0.full())
    cat_controls = np.array(u0[:, 0].full())

    modes = []
    states = []
    num_frames = 0
    for i in range(len(ref_states)): 
        u, X_pred = qc.solve(X0, u0, ref_states, i)
        
        cat_states = np.dstack((cat_states, np.array(X_pred.full())))
        cat_controls = np.dstack((cat_controls, np.array(u[:, 0].full())))
        
        t0, X0, u0 = qc.shift_timestep(t0, X_pred, u)
        qc.states = X0[:, 1]
        states.append(qc.states)
        modes.append(qc.mode)
        if qc.states[1] <= 0:
            break
        else:
            num_frames += 1

    np.save(np.array(modes), 'modes_along_wind.npy')
    np.save(np.array(states), 'states_along_wind.npy')
    simulate(ref_states, cat_states, num_frames, np.array([x_0, y_0, theta_0, x_goal, y_goal, theta_goal]), save=True)

if __name__ == "__main__":
    main()



    