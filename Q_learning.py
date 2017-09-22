import frozen_lake_data
import numpy as np
import plot

def run_q(env, g):

    # The inputs for the function are given in frozen_lake_data.py
    nrows = frozen_lake_data.nrows
    ncols = frozen_lake_data.ncols
    nactions = frozen_lake_data.nactions
    M = frozen_lake_data.M
    P = frozen_lake_data.P
    R = frozen_lake_data.R


    permanent_states = frozen_lake_data.permanent_states

    # Q_pi(s,a): Expected value, starting in state s of doing action a
    Q = [[0 for a in range(nactions)] for s in range(nrows * ncols)]

    # V_pi(s): Expected value in state s. Only used for plotting!
    V = [0 for s in range(nrows * ncols)]


    stable = False
    alpha = 0.2
    epsilon = 0.2

    total_rew = 0

    for i_episode in range(10000):

        s = env.reset()

        for t in range(1000):

            current_best_action = Q[s].index(max(Q[s]))

            V[s] = max(Q[s])

            action_prob = [epsilon / nactions for a in range(nactions)]
            action_prob[current_best_action] = 1 - epsilon + epsilon / nactions
            a = np.random.choice(nactions, 1, p=action_prob)[0]
            s_next, reward, done, info = env.step(a)

            Q[s][a] = Q[s][a] + alpha*(reward + g * max(Q[s_next]) - Q[s][a])

            s = s_next

            if reward == 1:
                total_rew += 1

            if s in permanent_states:
                #env.render()
                #print("Episode finished after {} timesteps".format(t + 1))
                break

    # Policy: Best action for every state
    pol = [0 for s in range(nrows * ncols)]

    for s in range(nrows * ncols):
        pol[s] = Q[s].index(max(Q[s]))

    plot_name = 'Q_learning_V'
    plot.plot_heat(V, nrows, ncols, plot_name)

    plot_name = 'Q_learning_pol'
    plot.plot_arrow(pol, nrows, ncols, plot_name)

    print('Q:')
    for row in range(4):
        print(Q[row * 4], Q[row * 4 + 1], Q[row * 4 + 2], Q[row * 4 + 3])

    print('pol:')
    for row in range(4):
        print(pol[row * 4], pol[row * 4 + 1], pol[row * 4 + 2], pol[row * 4 + 3])

    print(total_rew)




