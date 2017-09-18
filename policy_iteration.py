import frozen_lake_data

def policy_iteration(g):

    #The inputs for the function are given in frozen_lake_data.py
    nrows = frozen_lake_data.nrows
    ncols = frozen_lake_data.ncols
    nactions = frozen_lake_data.nactions
    M = frozen_lake_data.M
    P = frozen_lake_data.P
    R = frozen_lake_data.R

    # Policy: Best action for every state
    pol = [0 for s in range(nrows * ncols)]

    # V_pi(s): Expected value in state s
    V = [0 for s in range(nrows * ncols)]



    policy_stable = False

    while not policy_stable:
    #for i in range (1000):

        print('pol:')
        for row in range(4):
            print(pol[row * 4], pol[row * 4 + 1], pol[row * 4 + 2], pol[row * 4 + 3])

        # Policy evaluation
        theta = 0.0001
        delta = theta + 1

        while delta > theta:
            delta = 0
            for s in range(nrows * ncols):
                a = pol[s]
                v_temp = V[s]
                sum_PQ = 0
                for next_dir in range(4):
                    sum_PQ += P[s][a][next_dir] * V[M[s][next_dir]]
                V[s] = R[s][a] + g * sum_PQ

                delta = max(delta, abs(v_temp - V[s]))
            print(delta)



        # Policy improvement

        policy_stable = True
        for s in range(nrows * ncols):
            old_action = pol[s]
            Q_best = V[s]
            for a in range(nactions):
                sum_PQ = 0
                for next_dir in range(4):
                    sum_PQ += P[s][a][next_dir] * V[M[s][next_dir]]
                Q = R[s][a] + g * sum_PQ
                if Q > Q_best:
                    pol[s] = a
                    Q_best = Q
            if old_action != pol[s]:
                policy_stable = False

    print('V:')
    for row in range(4):
        print(V[row * 4], V[row * 4 + 1], V[row * 4 + 2], V[row * 4 + 3])

    return V, pol
