import frozen_lake_data


def value_iteration(g):

    #The inputs are given in frozen_lake_data.py
    nrows = frozen_lake_data.nrows
    ncols = frozen_lake_data.ncols
    nactions = frozen_lake_data.nactions
    M = frozen_lake_data.M
    P = frozen_lake_data.P
    R = frozen_lake_data.R


    # Q_pi(s,a): Expected value, starting in state s of doing action a
    Q = [[0 for a in range(nactions)] for s in range(nrows * ncols)]

    # V_pi(s): Expected value in state s
    V = [0 for s in range(nrows * ncols)]


    for i in range(1000):
        for row in range(nrows):
            for col in range(ncols):
                for a in range(4):
                    sum_PQ = 0
                    for next_dir in range(4):
                        sum_PQ += P[row*ncols + col][a][next_dir]*V[M[row*ncols + col][next_dir]]
                    Q[row*ncols + col][a] = R[row*ncols + col][a] + g*sum_PQ
                    V[row*ncols + col] = max(Q[row*ncols + col])

    print('Q:')
    for row in range(4):
        print(Q[row*4], Q[row*4 +1], Q[row*4 +2], Q[row*4 +3])

    print('V:')
    for row in range(4):
        print(V[row*4], V[row*4 +1], V[row*4 +2], V[row*4 +3])

    # Policy: Best action for every state
    pol = [0 for s in range(nrows * ncols)]

    for row in range(nrows):
        for col in range(ncols):
            pol[row*ncols + col] = Q[row*ncols + col].index(max(Q[row*ncols + col]))

    print('pol:')
    for row in range(4):
        print(pol[row * 4], pol[row * 4 + 1], pol[row * 4 + 2], pol[row * 4 + 3])

    return Q, pol


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

    not_changed = True

    # Q_pi(s,a): Expected value, starting in state s of doing action a
    Q = [[0 for a in range(nactions)] for s in range(nrows * ncols)]

    # V_pi(s): Expected value in state s
    V = [0 for s in range(nrows * ncols)]

