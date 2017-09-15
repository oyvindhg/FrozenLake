def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

#Using enumerators makes it easier to access the correct element in the matrices
d = enum('LEFT', 'DOWN', 'RIGHT', 'UP')