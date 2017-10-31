import gym
import policy_iteration
import value_iteration
import Q_learning
import policy_gradient_old
import policy_grad_cart
import policy_grad_pendulum
import policy_grad_cart_chapter
import policy_grad_cart_chapter2
import policy_gradient_cartpole
import policy_gradient_cartpole_baseline
import policy_gradient_pendulum
import policy_gradient_pendulum_continuous
import policy_gradient_frozenlake
import plot
import frozen_lake_data
import save_pendulum

if __name__ == "__main__":

    #Specify what we want to do
    problem = 'pendulum'
    method = 'pol_grad'
    run_simulation = False



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
            theta = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]

            T = [0 for s in range(len(theta))]
            I = [0 for s in range(len(theta))]

            for i in range(len(theta)):
                POL, V, t = value_iteration.run(env, S, A, M, P, R, gamma, theta[i])

                I[i] = t
                for s in range(nrows * ncols):
                    T[i] += V[s]
            print(len(theta))
            print(T)
            print(I)
            print(theta)

            plot.annotateplot(I, T, theta, 'Iterations', 'Total value', r'$\theta$', 'Iterations and sum of value function','IterValue')



        elif method == 'pol_iter':
            # gamma is the discount factor
            gamma = 0.95
            # theta is the convergence parameter
            theta = 0.001

            POL = policy_iteration.run(env, S, A, M, P, R, gamma, theta)




        elif method == 'Q_learning':
            # gamma is the discount factor
            gamma = 0.95
            # max_steps is the maximum number of steps allowed per episode
            max_steps = 200
            # n_episodes is the number of episodes run
            n_episodes = 10000

            methods = ['exp', 'opt']
            legends = [r'Varying $\epsilon$ and $\alpha$', 'Optimal policy']

            group_rewards_size = 100
            q_learning_rewards = [[0 for x in range(round(n_episodes * 2 / group_rewards_size))] for y in
                                  range(len(methods))]

            i = 0
            for mnum, method in enumerate(methods):

                rewards = []

                if method == 'opt':
                    n_episodes = 20000
                    POL = policy_iteration.run(env, S, A, M, P, R, gamma, 0.001)
                else:
                    POL, rewards = Q_learning.run(env, S, A, n_episodes, max_steps, gamma, method)

                plot_name = 'q_learning_POL_' + method
                plot.policy(POL, nrows, ncols, plot_name)


                for i in range(n_episodes):
                    s = env.reset()
                    t = 0
                    tot_reward = 0
                    while t < max_steps:
                        t += 1
                        action = POL[s]
                        s, reward, done, info = env.step(action)
                        tot_reward += reward
                        if done:
                            break
                    rewards.append(tot_reward)

                i = 0
                while((i+1)*group_rewards_size - 1 <= len(rewards)):
                    # print(i)
                    # print(q_learning_rewards[0][i])
                    # print(sum(rewards[i*group_rewards_size:(i+1)*group_rewards_size - 1]))

                    q_learning_rewards[mnum][i] = sum(rewards[i*group_rewards_size:(i+1)*group_rewards_size - 1])
                    i += 1


            plot.xyplot(range(i),q_learning_rewards, legends, 'Episodes (hundreds)','Reward','Q-learning performance','Q_learning')


        elif method == 'pol_grad':
            env = gym.make('FrozenLake-v0')

            # gamma is the discount factor
            gamma = 0.95

            learning_rate = 1e-2
            n_states = 16
            n_actions = 4
            hidden_layer_size = [8]

            total_episodes = 5000  # Set total number of episodes to train agent on.
            max_steps = 999
            ep_per_update = 5

            policy_gradient_frozenlake.run(env, learning_rate, n_states, n_actions, hidden_layer_size, total_episodes, max_steps,
                                ep_per_update, gamma)


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

    elif problem == 'cartpole':

        env = gym.make('CartPole-v0')

        # gamma is the discount factor
        gamma = 0.95

        learning_rate =1e-2
        n_states = 4
        n_actions = 2
        hidden_layer_size = [8]

        total_episodes = 2000  # Set total number of episodes to train agent on.
        max_steps = 999
        ep_per_update = 5

        rewards = policy_gradient_cartpole.run(env, learning_rate, n_states, n_actions, hidden_layer_size, total_episodes,
                                               max_steps, ep_per_update, gamma)


        plot.xyplot(range(len(rewards)), [rewards], 'nothing', 'Episodes (hundreds)', 'Reward',
                    'Policy gradient performance','Pol_gradient_cartpole_disc')

    elif problem == 'pendulum':

        env = gym.make('Pendulum-v0')

        # gamma is the discount factor
        gamma = 0.95

        learning_rate = 1e-2
        n_states = 3
        n_actions = 10
        hidden_layer_size = [8]

        total_episodes = 2000  # Set total number of episodes to train agent on.
        max_steps = 999
        ep_per_update = 5

        #rewards = policy_gradient_pendulum.run(env, learning_rate, n_states, n_actions, hidden_layer_size, total_episodes,
        #                                       max_steps, ep_per_update, gamma)

        rewards = policy_gradient_pendulum_continuous.run(env, learning_rate, n_states, -2, 2, hidden_layer_size,
                                                          total_episodes, max_steps, ep_per_update, gamma)

        plot.xyplot(range(len(rewards)), [rewards], 'nothing', 'Episodes (hundreds)', 'Reward',
                    'Policy gradient performance', 'Pol_gradient_pendulum_bl_10a')

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
