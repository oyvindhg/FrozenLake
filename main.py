import gym
import policy_iteration
import value_iteration
import Q_learning
import policy_gradient_cartpole
import policy_gradient_cartpole_baseline
import policy_gradient_pendulum
import policy_gradient_pendulum_continuous
import policy_gradient_frozenlake
import policy_gradient_frozenlake_baseline
import plot
import frozen_lake_data

if __name__ == "__main__":

    #Specify what we want to do
    problem = 'pendulum'
    method = 'pol_grad'
    investigate = "actions"

    run_simulation = False

    POL = [2 for s in range(16)]
    plot_name = 'pol_iter_pol_grad'
    plot.policy(POL, 4, 4, plot_name)

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
            gamma = 0.99

            learning_rate = 1e-2
            n_states = 16
            n_actions = 4
            hidden_layer_size = [20, 20]

            total_episodes = 5000  # Set total number of episodes to train agent on.
            max_steps = 999
            ep_per_update = 5

            policy_gradient_frozenlake_baseline.run(env, 1e-3, n_states, n_actions, hidden_layer_size, total_episodes, max_steps,
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
        max_steps = 200
        ep_per_update = 5
        group_size = 10
        avg_rewards = [[0 for x in range(round(total_episodes / group_size))] for y in
                                  range(0,3)]


        if investigate == "method":
            for i in range(0,3):
                if i == 0:
                    avg_rewards[0] = policy_gradient_cartpole.run(env, learning_rate, n_states, n_actions, hidden_layer_size, total_episodes,
                                                   max_steps, ep_per_update, gamma, False, group_size)
                elif i == 1:
                    avg_rewards[1] = policy_gradient_cartpole.run(env, learning_rate, n_states, n_actions, hidden_layer_size, total_episodes,
                                                   max_steps, ep_per_update, gamma, True, group_size)
                else:
                    avg_rewards[2] = policy_gradient_cartpole_baseline.run(env, learning_rate, n_states, n_actions, hidden_layer_size, total_episodes,
                                                   max_steps, ep_per_update, gamma, group_size)


            legends = ['Standard', 'Normalized', 'Using baseline']
            plot.xyplot(range(len(avg_rewards[0])), avg_rewards, legends, 'Episodes (tens)', 'Average reward',
                        'Policy gradient performance of different methods','Pol_gradient_cartpole_methods')
        elif investigate == "ep_per":
            for i in range(0, 3):
                if i == 0:
                    ep_per_update = 1
                    avg_rewards[0] = policy_gradient_cartpole_baseline.run(env, learning_rate, n_states, n_actions,
                                                                           hidden_layer_size, total_episodes,
                                                                           max_steps, ep_per_update, gamma,
                                                                           group_size)
                elif i == 1:
                    ep_per_update = 5
                    avg_rewards[1] = policy_gradient_cartpole_baseline.run(env, learning_rate, n_states, n_actions,
                                                                           hidden_layer_size, total_episodes,
                                                                           max_steps, ep_per_update, gamma,
                                                                           group_size)
                else:
                    ep_per_update = 20
                    avg_rewards[2] = policy_gradient_cartpole_baseline.run(env, learning_rate, n_states, n_actions,
                                                                           hidden_layer_size, total_episodes,
                                                                           max_steps, ep_per_update, gamma,
                                                                           group_size)
            legends = ['1 episode per update', '5 episodes per update', '20 episodes per update']
            plot.xyplot(range(len(avg_rewards[0])), avg_rewards, legends, 'Episodes (tens)', 'Average reward',
                        'Policy gradient performance of varying number of episodes per update', 'Pol_gradient_cartpole_ep')
        else:
            for i in range(0,3):
                if i == 0:
                    hidden_layer_size = [18]
                    avg_rewards[0] = policy_gradient_cartpole_baseline.run(env, learning_rate, n_states, n_actions, hidden_layer_size, total_episodes,
                                                   max_steps, ep_per_update, gamma, group_size)
                elif i == 1:
                    hidden_layer_size = [9, 9]
                    avg_rewards[1] = policy_gradient_cartpole_baseline.run(env, learning_rate, n_states, n_actions, hidden_layer_size, total_episodes,
                                                          max_steps, ep_per_update, gamma, group_size)
                else:
                    hidden_layer_size = [3, 3, 3, 3, 3, 3]
                    avg_rewards[2] = policy_gradient_cartpole_baseline.run(env, learning_rate, n_states, n_actions,
                                                                           hidden_layer_size, total_episodes,
                                                                           max_steps, ep_per_update, gamma, group_size)
            legends = ['One hidden layer', 'Two hidden layers', 'Six hidden layers']
            plot.xyplot(range(len(avg_rewards[0])), avg_rewards, legends, 'Episodes (tens)', 'Average reward',
                        'Policy gradient performance of varying network depth', 'Pol_gradient_cartpole_depth_2')


    elif problem == 'pendulum':

        env = gym.make('Pendulum-v0')


        # gamma is the discount factor
        gamma = 0.99

        learning_rate = 1e-2
        n_states = 3
        n_actions = 16
        hidden_layer_size = [16, 16]
        dropout_rate = 0.0

        total_episodes = 10000  # Set total number of episodes to train agent on.
        max_steps = 200
        ep_per_update = 5
        group_size = 10
        avg_rewards = [[0 for x in range(round(total_episodes / group_size))] for y in
                       range(0, 3)]



        if investigate == "dropout":
            for i in range(0, 3):
                if i == 0:
                    for j in range(0, 3):
                        dropout_rate = 0.0
                        current = policy_gradient_pendulum.run(env, learning_rate, n_states, n_actions, hidden_layer_size, dropout_rate, total_episodes,
                                               max_steps, ep_per_update, gamma, group_size)
                        for k in range(0,len(avg_rewards[0])):
                            avg_rewards[0][k] += current[k]
                elif i == 1:
                    for j in range(0, 3):
                        dropout_rate = 0.2
                        current = policy_gradient_pendulum.run(env, learning_rate, n_states, n_actions, hidden_layer_size, dropout_rate, total_episodes,
                                               max_steps, ep_per_update, gamma, group_size)
                        for k in range(0,len(avg_rewards[1])):
                            avg_rewards[1][k] += current[k]
                else:
                    for j in range(0, 3):
                        dropout_rate = 0.4
                        current = policy_gradient_pendulum.run(env, learning_rate, n_states, n_actions, hidden_layer_size, dropout_rate, total_episodes,
                                               max_steps, ep_per_update, gamma, group_size)
                        for k in range(0,len(avg_rewards[2])):
                            avg_rewards[2][k] += current[k]

            for i in range(3):
                for j in range(len(avg_rewards[0])):
                    avg_rewards[i][j] /= 3
            legends = ['No dropout', '20% dropout', '40% dropout']
            plot.xyplot(range(len(avg_rewards[0])), avg_rewards, legends, 'Episodes (tens)', 'Average reward',
                        'Policy gradient performance with varying dropout rate', 'Pol_gradient_pendulum_dropout')

        elif investigate == "actions":
            for i in range(0, 3):
                if i == 0:
                    for j in range(0,3):
                        n_actions = 4
                        current = policy_gradient_pendulum.run(env, learning_rate, n_states, n_actions, hidden_layer_size, dropout_rate, total_episodes,
                                               max_steps, ep_per_update, gamma, group_size)
                        for k in range(0,len(avg_rewards[0])):
                            avg_rewards[0][k] += current[k]
                elif i == 1:
                    for j in range(0,3):
                        n_actions = 16
                        current = policy_gradient_pendulum.run(env, learning_rate, n_states, n_actions, hidden_layer_size, dropout_rate, total_episodes,
                                               max_steps, ep_per_update, gamma, group_size)
                        for k in range(0,len(avg_rewards[1])):
                            avg_rewards[1][k] += current[k]
                else:
                    for j in range(0,3):
                        n_actions = 24
                        current = policy_gradient_pendulum.run(env, learning_rate, n_states, n_actions, hidden_layer_size, dropout_rate, total_episodes,
                                               max_steps, ep_per_update, gamma, group_size)
                        for k in range(0,len(avg_rewards[2])):
                            avg_rewards[2][k] += current[k]
            for i in range(3):
                for j in range(len(avg_rewards[0])):
                    avg_rewards[i][j] /= 3

            legends = ['4 actions', '16 actions', '24 actions']
            plot.xyplot(range(len(avg_rewards[0])), avg_rewards, legends, 'Episodes (tens)', 'Average reward',
                        'Policy gradient performance with varying number of actions', 'Pol_gradient_pendulum_actions')
        elif investigate == "hl":
            for i in range(0,3):
                if i == 0:
                    for j in range(0,3):
                        hidden_layer_size = [32]
                        current = policy_gradient_pendulum.run(env, learning_rate, n_states, n_actions,
                                                                               hidden_layer_size, dropout_rate, total_episodes,
                                                                               max_steps, ep_per_update, gamma, group_size)
                        for k in range(0, len(avg_rewards[0])):
                            avg_rewards[0][k] += current[k]
                elif i == 1:
                    for j in range(0,3):
                        hidden_layer_size = [16, 16]
                        current = policy_gradient_pendulum.run(env, learning_rate, n_states, n_actions,
                                                                               hidden_layer_size, dropout_rate, total_episodes,
                                                                               max_steps, ep_per_update, gamma, group_size)
                        for k in range(0, len(avg_rewards[1])):
                            avg_rewards[1][k] += current[k]
                else:
                    for j in range(0,3):
                        hidden_layer_size = [8, 8, 8, 8]
                        current = policy_gradient_pendulum.run(env, learning_rate, n_states, n_actions,
                                                                               hidden_layer_size, dropout_rate, total_episodes,
                                                                               max_steps, ep_per_update, gamma, group_size)
                        for k in range(0, len(avg_rewards[2])):
                            avg_rewards[2][k] += current[k]

            for i in range(3):
                for j in range(len(avg_rewards[0])):
                    avg_rewards[i][j] /= 3

            legends = ['One hidden layer', 'Two hidden layers', 'Four hidden layers']
            plot.xyplot(range(len(avg_rewards[0])), avg_rewards, legends, 'Episodes (tens)', 'Average reward',
                        'Policy gradient performance of varying network depth', 'Pol_gradient_pendulum_depth')
        else:
            rewards = policy_gradient_pendulum.run(env, learning_rate, n_states, n_actions, hidden_layer_size, dropout_rate, total_episodes,
                                               max_steps, ep_per_update, gamma, group_size)

            plot.xyplot(range(len(rewards)), [rewards], 'nothing', 'Episodes (tens)', 'Average reward',
                       'Policy gradient performance', 'Pol_gradient_pendulum')


            # rewards = policy_gradient_pendulum_continuous.run(env, learning_rate, n_states, -2, 2, hidden_layer_size, dropout_rate,
            #                                              total_episodes, max_steps, ep_per_update, gamma)



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
