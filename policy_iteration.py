import plot
import random

def optimize(env, S, A, M, P, R, gamma, theta):

    nactions = A
    nrows = S[0]
    ncols = S[1]

    # Policy: Best action for every state. Randomize the policy at first
    POL = [0 for s in range(nrows * ncols)]
    for s in range(nrows * ncols):
        POL[s] = random.randint(0, nactions - 1)

    #V_pi(s): Expected value in state s
    V = [0 for s in range(nrows * ncols)]

    # Q_pi(s,a): Expected value, starting in state s of doing action a
    Q = [[0 for a in range(nactions)] for s in range(nrows * ncols)]

    policy_stable = False

    count = 0
    while not policy_stable:
        count += 1

        if count in range(4):
            plot_name = 'pol_iter_POL_' + str(count-1)
            plot.policy(POL, nrows, ncols, plot_name)


        # Policy evaluation
        delta = theta + 1
        while delta > theta:
            delta = 0
            for s in range(nrows * ncols):
                a = POL[s]
                V_prev = V[s]
                PV = 0
                for next_dir in range(4):
                    PV += P[s][a][next_dir] * V[M[s][next_dir]]
                V[s] = R[s][a] + gamma * PV

                delta = max(delta, abs(V_prev - V[s]))


        # Policy improvement
        policy_stable = True
        for s in range(nrows * ncols):

            POL_prev = POL[s]
            Q_best = V[s]
            for a in range(nactions):
                PV = 0
                for next_dir in range(4):
                    PV += P[s][a][next_dir] * V[M[s][next_dir]]
                Q[s][a] = R[s][a] + gamma * PV
                if Q[s][a] > Q_best:
                    POL[s] = a
                    Q_best = Q[s][a]
            if POL_prev != POL[s]:
                policy_stable = False

        if count in range(4):
            plot_name = 'pol_iter_V_' + str(count)
            plot.heatplot(V, nrows, ncols, plot_name)

        # print('POL:')
        # for row in range(4):
        #     print(POL[row * 4], POL[row * 4 + 1], POL[row * 4 + 2], POL[row * 4 + 3])
        #
        # print('V:')
        # for row in range(4):
        #     print(V[row * 4], V[row * 4 + 1], V[row * 4 + 2], V[row * 4 + 3])
        #
        # print('Q:')
        # for row in range(4):
        #     print(Q[row * 4], Q[row * 4 + 1], Q[row * 4 + 2], Q[row * 4 + 3])



    #print('Nr of iterations: ', count)


    return POL
