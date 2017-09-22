import matplotlib.pylab as plt

def plot_heat(S, nrows, ncols, name):

    M = [[0 for r in range(nrows)] for s in range(ncols)]

    for row in range(nrows):
        for col in range(ncols):
            M[row][col] = S[row*ncols + col]

    fig, ax = plt.subplots()

    ax.matshow(M, cmap=plt.cm.Blues)

    for i in range(nrows):
        for j in range(ncols):
            c = M[j][i]
            c = round(c,2)
            ax.text(i, j, str(c), va='center', ha='center')

    my_path = '/home/oyvindhg/PycharmProjects/FrozenLake/Plots/'
    full_name = name + '.png'
    plt.savefig(my_path + full_name, bbox_inches='tight')


    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_aspect('equal')
    # plt.imshow(M, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.colorbar()
    #
    # labels = range(0, len(M[0]))
    # plt.xticks(labels)
    #
    # labels = range(0, len(M))
    # plt.yticks(labels)
    #
    # plt.show()

def plot_arrow(S, nrows, ncols, name):

    W = [[0 for r in range(nrows)] for s in range(ncols)]
    M = [[0 for r in range(nrows)] for s in range(ncols)]

    for row in range(nrows):
        for col in range(ncols):
            M[row][col] = S[row * ncols + col]

    fig, ax = plt.subplots()

    ax.matshow(W, cmap=plt.cm.Blues)

    for row in range(nrows):
        for col in range(ncols):
            x = 0
            y = 0

            if M[row][col] == 0:
                x = -1
            elif M[row][col] == 1:
                y = -1
            elif M[row][col] == 2:
                x = 1
            else:
                y = 1
            ax.quiver(col, row, x, y)

    labels = range(0, len(M[0]))
    plt.xticks(labels)

    labels = range(0, len(M))
    plt.yticks(labels)

    my_path = '/home/oyvindhg/PycharmProjects/FrozenLake/Plots/'
    full_name = name + '.png'
    plt.savefig(my_path + full_name, bbox_inches='tight')

def plot_board(nrows, ncols, start, holes, goal):

    M = [[0 for r in range(nrows)] for s in range(ncols)]

    for row in range(nrows):
        for col in range(ncols):
            if row*nrows + col == start:
                M[row][col] = 1
            elif row*nrows + col in holes:
                M[row][col] = 3
            elif row * nrows + col == goal:
                M[row][col] = 2

    fig, ax = plt.subplots()

    ax.matshow(M, cmap=plt.cm.Blues)

    for row in range(nrows):
        for col in range(ncols):
            if row*nrows + col == start:
                ax.text(col, row, 'Start', va='center', ha='center')
            elif row*nrows + col in holes:
                ax.text(col, row, 'Hole', va='center', ha='center')
            elif row * nrows + col == goal:
                ax.text(col, row, 'Goal', va='center', ha='center')

    plt.savefig('board.png', bbox_inches='tight')