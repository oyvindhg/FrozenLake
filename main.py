import gym
import policy_iteration
import value_iteration
import Q_learning
import policy_gradient
import policy_grad_cart
import policy_grad_pendulum
import plot
import frozen_lake_data
import save_pendulum

if __name__ == "__main__":

    #Specify what we want to do
    problem = 'frozen_lake'
    method = 'pol_iter'
    run_simulation = True



    if problem == 'frozen_lake':
        env = gym.make('FrozenLake-v0')

        # Import the Frozen Lake data
        nrows = frozen_lake_data.nrows
        ncols = frozen_lake_data.ncols
        S = [nrows, ncols]
        A = frozen_lake_data.nactions
        M = frozen_lake_data.M  # Matrix to show final state when doing an action from an initial state
        P = frozen_lake_data.P  # P(s'|s,a)
        R = frozen_lake_data.R  # R(s,a)
        start = frozen_lake_data.start
        goal = frozen_lake_data.goal
        permanent_states = frozen_lake_data.permanent_states
        POL = [0 for s in range(nrows * ncols)]

        if method == 'print_board':
            plot.frozen_lake_board(nrows, ncols, start, permanent_states, goal)

        elif method == 'val_iter':
            # gamma is the discount factor
            gamma = 0.95
            # theta is the convergence parameter
            theta = 0.0001

            POL = value_iteration.optimize(env, S, A, M, P, R, gamma, theta)

        elif method == 'pol_iter':
            # gamma is the discount factor
            gamma = 0.95
            # theta is the convergence parameter
            theta = 0.001

            POL = policy_iteration.optimize(env, S, A, M, P, R, gamma, theta)

        if run_simulation:
            s = env.reset()
            max_steps = 200
            t = 0
            while t < max_steps:
                t += 1
                env.render()
                action = POL[s]
                s, reward, done, info = env.step(action)
                if done:
                    break
            print(s)
            env.render()
            print("Episode finished after {} timesteps".format(t + 1))






    #env = gym.make('CartPole-v0')
    #env = gym.make('Pendulum-v0')

    #policy_gradient.run_pg(env, g)
    #policy_grad_pendulum.run_pg(env, g)
    #save_pendulum.run_pg(env, g)

    # for i_episode in range(1):
    #     s = env.reset()
    #     t = 0
    #     for steps in range(1000):
    #         t += 1
    #         #env.render()
    #         action = [-1.6]
    #         s, reward, done, info = env.step(action)
    #         env.render()

    #Q_learning.run_q(env, g)
    #policy_iteration.run_pi(env, g)
    #value_iteration.run_vi(env, g)
