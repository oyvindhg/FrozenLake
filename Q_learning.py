import frozen_lake_data
import numpy as np
import plot
import math

def set_alpha_epsilon(ep, n_episodes, method):

    if method == 'none_high':
        alpha = 0.1
        epsilon = 0.1

    elif method == 'alpha_high':
        alpha = 0.1
        epsilon = 0.1

    elif method == 'epsilon_high':
        alpha = 0.1
        epsilon = 0.5
    elif method == 'both_high':
        alpha = 0.3
        epsilon = 0.5
    elif method == 'exp':
        alpha = max(math.exp(-ep/(n_episodes*0.05)),0.1)
        epsilon = max(math.exp(-ep/(n_episodes*0.05)),0.1)
    else:
        raise ValueError("Not a valid method for setting alpha and epsilon")

    return alpha, epsilon




def run(env, S, A, n_episodes, max_steps, gamma, method):

    nstates = S[0]*S[1]
    nactions = A

    # Q_pi(s,a): Expected value, starting in state s of doing action a
    Q = [[0 for a in range(nactions)] for s in range(nstates)]

    # V_pi(s): Expected value in state s. Only used for plotting!
    V = [0 for s in range(nstates)]

    rewards = []

    for ep in range(n_episodes):

        rewards.append(0)

        s = env.reset()

        for step in range(max_steps):

            alpha, epsilon = set_alpha_epsilon(ep, n_episodes, method)

            current_best_action = Q[s].index(max(Q[s]))

            V[s] = max(Q[s])

            action_prob = [epsilon / nactions for a in range(nactions)]
            action_prob[current_best_action] = 1 - epsilon + epsilon / nactions
            a = np.random.choice(nactions, 1, p=action_prob)[0]
            s_next, reward, done, info = env.step(a)

            Q[s][a] = Q[s][a] + alpha*(reward + gamma * max(Q[s_next]) - Q[s][a])

            s = s_next

            rewards[ep] += reward

            if done:
                break

    # Policy: Best action for every state
    POL = [0 for s in range(nstates)]

    for s in range(nstates):
        POL[s] = Q[s].index(max(Q[s]))

    # print('Q:')
    # for row in range(4):
    #     print(Q[row * 4], Q[row * 4 + 1], Q[row * 4 + 2], Q[row * 4 + 3])
    #
    # print('pol:')
    # for row in range(4):
    #     print(pol[row * 4], pol[row * 4 + 1], pol[row * 4 + 2], pol[row * 4 + 3])

    return POL, rewards




