import enumerate

nrows = 4
ncols = 4
nactions = 4

d = enumerate.d
permanent_states = [5,7,11,12,15]

# Matrix to show which state you end up in by moving in a given direction from any state
M =  [[0 for m in [d.LEFT, d.DOWN, d.RIGHT, d.UP]] for s in range(nrows*ncols)]
for row in range(nrows):
    for col in range(ncols):
        M[row * ncols + col][d.LEFT] = row * ncols + max(col - 1, 0)
        M[row * ncols + col][d.DOWN] = min(row + 1, nrows - 1) * ncols + col
        M[row * ncols + col][d.RIGHT] = row * ncols + min(col + 1, ncols - 1)
        M[row * ncols + col][d.UP] = max(row - 1, 0) * ncols + col


# R(s,a,s'): Immediate reward for starting in state s, doing action a, and ending up in state s'
R_sas = [[[0 for dir_next in [d.LEFT, d.DOWN, d.RIGHT, d.UP]] for a in [d.LEFT, d.DOWN, d.RIGHT, d.UP]] for s in range(nrows*ncols)]
for a in range(4):
    R_sas[nrows * (ncols - 1) + ncols - 2][a][d.RIGHT] = 1
    R_sas[nrows * (ncols - 2) + ncols - 1][a][d.DOWN] = 1


# P(s'|s,a): Probability of ending up in state s' from state s by doing action a
P = [[[0 for dir_next in [d.LEFT, d.DOWN, d.RIGHT, d.UP]] for a in [d.LEFT, d.DOWN, d.RIGHT, d.UP]] for s in range(nrows*ncols)]
for row in range(nrows):
    for col in range(ncols):
        if not row*ncols + col in permanent_states:
            for a in range(nactions):
                P[row * ncols + col][a][(a - 1) % 4] = 1 / 3
                P[row * ncols + col][a][a] = 1 / 3
                P[row * ncols + col][a][(a + 1) % 4] = 1 / 3


# R(s,a) = sum over s' of P(s'|s,a)R(s,a,s')
R = [[0 for a in [d.LEFT, d.DOWN, d.RIGHT, d.UP]] for s in range(nrows * ncols)]
for a in range(nactions):
    for dir_next in range(4):
        R[nrows * (ncols - 1) + ncols - 2][a] += R_sas[nrows * (ncols - 1) + ncols - 2][a][dir_next] * P[nrows * (ncols - 1) + ncols - 2][a][dir_next]
        R[nrows * (ncols - 2) + ncols - 1][a] += R_sas[nrows * (ncols - 2) + ncols - 1][a][dir_next] * P[nrows * (ncols - 2) + ncols - 1][a][dir_next]
