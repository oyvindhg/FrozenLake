import plot


def optimize(env, S, A, M, P, R, gamma, theta):

    nactions = A
    nrows = S[0]
    ncols = S[1]

    # Q_pi(s,a): Expected value, starting in state s of doing action a
    Q = [[0 for a in range(nactions)] for s in range(nrows * ncols)]

    # V_pi(s): Expected value in state s
    V = [0 for s in range(nrows * ncols)]

    delta = theta + 1
    count = 0
    while delta > theta:
        count += 1
        delta = 0
        for s in range(nrows * ncols):
            V_prev = V[s]
            for a in range(4):
                PV = 0
                for next_dir in range(4):
                    PV += P[s][a][next_dir]*V[M[s][next_dir]]
                Q[s][a] = R[s][a] + gamma*PV
            V[s] = max(Q[s])
            delta = max(delta, abs(V_prev - V[s]))

        if count in range(4):
            plot_name = 'val_iter_V_' + str(count)
            plot.heatplot(V, nrows, ncols, plot_name)

        # print('V:')
        # for row in range(4):
        #     print(V[row * 4], V[row * 4 + 1], V[row * 4 + 2], V[row * 4 + 3])
        #
        # print('Q:')
        # for row in range(4):
        #     print(Q[row * 4], Q[row * 4 + 1], Q[row * 4 + 2], Q[row * 4 + 3])

    plot_name = 'val_iter_V_' + str(count)
    plot.heatplot(V, nrows, ncols, plot_name)

    #print('Nr of iterations: ', count)

    # Policy: Best action for every state
    POL = [0 for s in range(nrows * ncols)]

    for s in range(nrows * ncols):
        if max(Q[s]) == 0:
            POL[s] = -1
        else:
            POL[s] = Q[s].index(max(Q[s]))

    plot.policy(POL, nrows, ncols, 'val_iter_POL')

    print('POL:')
    for row in range(4):
        print(POL[row * 4], POL[row * 4 + 1], POL[row * 4 + 2], POL[row * 4 + 3])

    return POL
