def relation(i, j):
    vertical = 20
    vertical_rem = 19
    if i == j:
        return 0
    if i % vertical != 0 and i % vertical != vertical_rem:
        if i+1 == j or i-1 == j or i+vertical == j or i-vertical == j:
            return 1

        else:
            return 0
    elif i % vertical == 0:
        if i+1 == j or i+vertical == j or i-vertical == j:
            return 1

        else:
            return 0
    elif i % vertical == vertical_rem:
        if i-1 == j or i+vertical == j or i-vertical == j:
            return 1

        else:
            return 0


row_col_size = 20
size = row_col_size*row_col_size
adjacencey_mat = [[0 for i in range(size)] for j in range(size)]

for i in range(0, size):
    for j in range(0, size):
        result = relation(i, j)
        adjacencey_mat[i][j] = result


def return_matrix():
    return adjacencey_mat, size
