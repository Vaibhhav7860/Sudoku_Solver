# Taking size = 9 for a 9x9 matrix
SIZE = 9
# Taking an empty list named matrix which will be created as a
# 2-D array of different rows which we'll take as input.
matrix = []

for i in range(9):
    # Taking 9 rows as input and appending it into empty matrix.
    l = list(map(int,input().split()))
    matrix.append(l)


# Function for returning the output matrix.
def return_sudoku(matrix):
    return matrix


# A function for checking the unassigned cells of the matrix
# and referring its respective row and coloumn.
def number_unassigned(row, col):
    num_unassign = 0
    for i in range(0,SIZE):
        for j in range (0,SIZE):

            if matrix[i][j] == 0:
                row = i
                col = j
            # If we find the empty cell, updating the value of num_unassign to 1
            # as we start inserting number from 1 and it doesn't fit then we move ahead
                num_unassign = 1
            # Creating a list having the referred row, column and the updated value of num_unassign
                a = [row, col, num_unassign]
                return a
    # If no empty cell is found then we assign -1 to row and column and num_unassign remains 0.
    row = -1
    col = -1
    a = [row, col, num_unassign]
    return a

def is_safe(n, r, c):
# This for loop iterates to each block of each row and checks if a number(1-9)
# is already present in the cell. If the number is present it returns false which means the cell
# is not safe to insert any digit.
    for i in range(0,SIZE):

        if matrix[r][i] == n:
            return False
# Similarly this for loop iterates for each column and inside each column
# iterates each cell and performs the same task.

    for i in range(0,SIZE):

        if matrix[i][c] == n:
            return False

# Considering each 3x3 sub-matrix and initializing its rows and columns
    row_start = (r//3)*3
    col_start = (c//3)*3

# Checking for each cell of that sub-matrix and identifying
# whether the block is safe to insert or not.

    for i in range(row_start,row_start+3):
        for j in range(col_start,col_start+3):
            if matrix[i][j]==n:
                return False
# After performing all the 3 searches if the number is not found then
# we'll return true which means the cell is safe to insert a number
    return True


def solve_sudoku(matrix):
    row = 0
    col = 0

    a = number_unassigned(row, col)
    if a[2] == 0:
        # It means the unassigned_num is 0 that means the referrred block is already filled,
        # So, we don't need to change its value as it is a default value therefore we
        # directly return true.
        return True
    # If the above condition is false therefore we get an empty box
    # So, assigning the row index and column index into their respective variables row and col.
    row = a[0]
    col = a[1]

    for i in range(1,10):
        # now if the number i is not present in either that row, column and
        # the sub-matrix of the referred cell we assign that cell with the given number i.
        if is_safe(i, row, col):
            matrix[row][col] = i
        # After updating the cell value we recursively check for the updated matrix
            if solve_sudoku(matrix):
                # If we can't update the matrix further that means our puzzle is solved so we return true.
                return True
            # Else if any value is violating the constraint we backtrack and re-assign
            # that cell to 0 and return false.
            matrix[row][col]=0
    return False

if solve_sudoku(matrix):
    return_sudoku(matrix)
else:
    print("No solution")


'''
x = solve_sudoku(matrix)
if x:
    y = print_sudoku(matrix)
    print(y)
else:
    print("No Solution !!!")
'''