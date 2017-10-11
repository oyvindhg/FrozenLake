import matplotlib.pylab as plt
import numpy as np
import os

def annotateplot(x,y,n,xlabel, ylabel, nlabel, plotname, fname):

    fig, ax = plt.subplots()

    ax.scatter(x, y)
    q = ax.plot(x, y, label=nlabel)


    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    plt.title(plotname)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend(handles=q)

    cwd = os.getcwd()
    full_name = fname + '.eps'
    plt.savefig(cwd + '/Plots/' + full_name, bbox_inches='tight')

def xyplot(x, y, legends, xlabel, ylabel, plotname, name):

    fig, ax = plt.subplots()

    for i in range(len(y)):
        plt.plot(x, y[i], label=legends[i])
        plt.legend()

    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(0, end, 10))

    plt.title(plotname)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cwd = os.getcwd()
    full_name = name + '.eps'
    plt.savefig(cwd + '/Plots/' + full_name, bbox_inches='tight')


def heatplot(S, nrows, ncols, name):

    M = [[0 for r in range(nrows)] for s in range(ncols)]

    for row in range(nrows):
        for col in range(ncols):
            M[row][col] = S[row*ncols + col]

    fig, ax = plt.subplots()

    ax.set_xticks(np.arange(-.5, 10, 1), minor=True);
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True);

    ax.matshow(M, vmin=0, vmax = 1, cmap=plt.cm.Blues)

    ax.tick_params(which='minor', axis='both', left='off', top='off', right='off', bottom='off', labelleft='off',
                   labeltop='off',
                   labelright='off', labelbottom='off')

    ax.tick_params(which='major', axis='both', left='off', top='off', right='off', bottom='off', labelleft='off',
                   labeltop='off',
                   labelright='off', labelbottom='off')

    for i in range(nrows):
        for j in range(ncols):
            c = M[j][i]
            c = round(c,2)
            ax.text(i, j, str(c), va='center', ha='center')

    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

    cwd = os.getcwd()
    full_name = name + '.eps'
    plt.savefig(cwd + '/Plots/' + full_name, bbox_inches='tight')


def policy(S, nrows, ncols, name):

    W = [[0 for r in range(nrows)] for s in range(ncols)]
    M = [[0 for r in range(nrows)] for s in range(ncols)]

    for row in range(nrows):
        for col in range(ncols):
            M[row][col] = S[row * ncols + col]

    fig, ax = plt.subplots()

    ax.set_xticks(np.arange(-.5, 10, 1), minor=True);
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True);

    ax.matshow(W, cmap=plt.cm.Blues)

    ax.tick_params(which='minor', axis='both', left='off', top='off', right='off', bottom='off', labelleft='off',
                   labeltop='off',
                   labelright='off', labelbottom='off')

    ax.tick_params(which='major', axis='both', left='off', top='off', right='off', bottom='off', labelleft='off',
                   labeltop='off',
                   labelright='off', labelbottom='off')

    for row in range(nrows):
        for col in range(ncols):
            x = 0
            y = 0

            if row*nrows+col in [5,7,11,12,15]:
                M[row][col] = -1

            if M[row][col] == 0:
                x = -1
            elif M[row][col] == 1:
                y = -1
            elif M[row][col] == 2:
                x = 1
            elif M[row][col] == 3:
                y = 1
            else:
                continue
            ax.quiver(col, row, x, y, headaxislength=5, scale=nrows*ncols-2)

    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

    cwd = os.getcwd()
    full_name = name + '.eps'
    plt.savefig(cwd + '/Plots/' + full_name, bbox_inches='tight')

def frozen_lake_board(nrows, ncols, start, holes, goal):

    M = [[0 for r in range(nrows)] for s in range(ncols)]

    for row in range(nrows):
        for col in range(ncols):
            if row*nrows + col == start:
                M[row][col] = 1
            elif row * nrows + col == goal:
                M[row][col] = 1
            elif row*nrows + col in holes:
                M[row][col] = 3
            else:
                M[row][col] = 0

    fig, ax = plt.subplots()

    ax.set_xticks(np.arange(-.5, 10, 1), minor=True);
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True);


    ax.matshow(M, vmin=0, vmax = 10, cmap=plt.cm.Blues)

    ax.tick_params(which='minor', axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',
                   labelright='off', labelbottom='off')

    ax.tick_params(which='major', axis='both', left='off', top='off', right='off', bottom='off', labelleft='off',
                   labeltop='off',
                   labelright='off', labelbottom='off')

    for row in range(nrows):
        for col in range(ncols):
            if row*nrows + col == start:
                ax.text(col, row, 'Start', va='center', ha='center')
            elif row*nrows + col in holes:
                ax.text(col, row, 'Hole', va='center', ha='center')
            elif row * nrows + col == goal:
                ax.text(col, row, 'Goal', va='center', ha='center')

    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

    cwd = os.getcwd()

    plt.savefig(cwd + '/Plots/board.eps', bbox_inches='tight')