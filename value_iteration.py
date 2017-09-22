import frozen_lake_data
import plot


def run_vi(env, g):

    #The inputs are given in frozen_lake_data.py
    nrows = frozen_lake_data.nrows
    ncols = frozen_lake_data.ncols
    nactions = frozen_lake_data.nactions
    M = frozen_lake_data.M
    P = frozen_lake_data.P
    R = frozen_lake_data.R


    permanent_states = frozen_lake_data.permanent_states

    # Q_pi(s,a): Expected value, starting in state s of doing action a
    Q = [[0 for a in range(nactions)] for s in range(nrows * ncols)]

    # V_pi(s): Expected value in state s
    V = [0 for s in range(nrows * ncols)]


    theta = 0.0001
    delta = theta + 1

    count = 0
    while delta > theta:
        count += 1
        delta = 0
        for s in range(nrows * ncols):
            v_temp = V[s]
            for a in range(4):
                sum_PQ = 0
                for next_dir in range(4):
                    sum_PQ += P[s][a][next_dir]*V[M[s][next_dir]]
                Q[s][a] = R[s][a] + g*sum_PQ
            V[s] = max(Q[s])
            delta = max(delta, abs(v_temp - V[s]))

        if count in range(4):
            plot_name = 'val_it_' + str(count)
            plot.plot_heat(V, nrows, ncols, plot_name)

        print('V:')
        for row in range(4):
            print(V[row * 4], V[row * 4 + 1], V[row * 4 + 2], V[row * 4 + 3])

        print('Q:')
        for row in range(4):
            print(Q[row * 4], Q[row * 4 + 1], Q[row * 4 + 2], Q[row * 4 + 3])

    plot_name = 'val_it_' + str(count)
    plot.plot_heat(V, nrows, ncols, plot_name)

    print('Nr of iterations: ', count)


    # Policy: Best action for every state
    pol = [0 for s in range(nrows * ncols)]

    for s in range(nrows * ncols):
        pol[s] = Q[s].index(max(Q[s]))

    #plot.plot_arrow(pol, nrows, ncols)

    print('pol:')
    for row in range(4):
        print(pol[row * 4], pol[row * 4 + 1], pol[row * 4 + 2], pol[row * 4 + 3])

    for i_episode in range(1):
        s = env.reset()
        t = 0
        while True:
            t += 1
            #env.render()
            action = pol[s]
            s, reward, done, info = env.step(action)
            if s in permanent_states:
                env.render()
                print("Episode finished after {} timesteps".format(t + 1))
                break
