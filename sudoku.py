import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

'''
README: The same figure is used for both the plots, so to separate them out,
please comment out 239 -243 to view the S_hardplot and solution.
Vice versa, comment out 247-252 for the S_easy.
'''

def print_sudoku(S, constraint_matrix, colour1, colour2):
    print(constraint_matrix[0][0])
    for i in range(S.shape[0]):
        if i %3 == 0:
            sys.stdout.write('----------------------\n')
        for j in range(S.shape[1]):
            if j % 3 == 0:
                sys.stdout.write('|')
            if constraint_matrix[i][j] == 0:
              sys.stdout.write(colour1 + str(S[i,j]))
            else:
              sys.stdout.write(colour2 + str(S[i,j]))
            sys.stdout.write(' ')
        sys.stdout.write('|')

        sys.stdout.write('\n')

def sudoku_to_one_hot(S):
    # initialize with zeros
    x_vol = np.zeros((9, 9, 9))

    # generate meshgrid to assign value for every position
    xx, yy = np.meshgrid(np.arange(9), np.arange(9))
    xx = xx[S > 0]
    yy = yy[S > 0]
    ss = S[S > 0]

    # -1 because sudoku starts at 1 and vol starts at 0
    x_vol[yy, xx, ss - 1] = 1
    return x_vol

def sudoku_to_x(S):
    return sudoku_to_one_hot(S).ravel()

def one_hot_to_sudoku(x_vol):
    S = x_vol.argmax(axis=-1) + 1
    S[x_vol.sum(axis=-1) < 1] = 0
    return S

def x_to_sudoku(x):
    return one_hot_to_sudoku(x.reshape(9, 9, 9))

def get_Arow():
    # row constraints
    I9 = sp.eye(9)
    Ir = sp.bmat([[I9, I9, I9, I9, I9, I9, I9, I9, I9]])

    A_row = sp.block_diag([Ir, Ir, Ir, Ir, Ir, Ir, Ir, Ir, Ir])
    return A_row

def get_Acol():
    I81 = sp.eye(81)
    Acol = sp.bmat([[I81, I81, I81, I81, I81, I81, I81, I81, I81]])
    return Acol

def get_Abox():
    I9 = sp.eye(9)
    I9x27 = sp.bmat([[I9, I9, I9]])
    Ab = sp.block_diag([I9x27, I9x27, I9x27])
    Ab1 = sp.bmat([[Ab, Ab, Ab]])
    Abox = sp.block_diag([Ab1, Ab1, Ab1])
    return Abox

def get_Acell():
    one_9 = np.ones(9).T
    Ic = sp.block_diag([one_9, one_9, one_9, one_9, one_9, one_9, one_9, one_9, one_9])
    Acell = sp.block_diag([Ic, Ic, Ic, Ic, Ic, Ic, Ic, Ic, Ic])
    return Acell

def get_Aclue(sudoku, possible_values):
    Aclue = []
    row, col = sudoku.shape
    for r in range(row):
        for c in range(col):
            if sudoku[r, c] != 0:
                clue_row = np.zeros(729)
                clue_row[r*81 + c*9 + sudoku[r, c] - 1] = 1.
                Aclue.append(clue_row)
    return np.array(Aclue)

# def get_Aclue(sudoku, possible_values):
#     row, col = sudoku.shape
#     non_zero_indices = np.flatnonzero(sudoku)
#     #print(non_zero_indices)
#     n_clues = sudoku[np.nonzero(sudoku)]
#     #print(n_clues - 1)
#     Aclue = np.zeros((row*col*possible_values, n_clues.size))
#     i = 0
#     for index, value in enumerate(non_zero_indices):
#       Aclue[value*possible_values + n_clues[i]][index] = 1
#     return Aclue.T

def get_A(sudoku, possible_values):
    a = sp.vstack([get_Arow(), get_Acol(), get_Abox(), get_Acell(), get_Aclue(sudoku, possible_values)])
    #a = np.bmat([get_Arow(), get_Acol(), get_Abox(), get_Acell(), get_Aclue(sudoku, possible_values)])
    return a

def dual_affine_scaling(A):

  # Initialise invariant constraints
  constraints = A.shape[0]
  c = np.ones(A.shape[1]).reshape(A.shape[1], 1)
  yk = np.zeros(constraints).reshape(constraints, 1)
  b = np.ones(constraints).reshape(constraints, 1)
  i = 0
  iteration_condition = 1

  # plotting variables
  i_list = []
  y_list = []
  i_list.append(i)
  y_list.append(b.T@yk)
  while iteration_condition:
    i += 1
    # Find sk from inequality
    sk = (c - A.T@yk)
    #print(i, sk)

    # Construct Hk matrix
    sk_power_2 = np.power(sk, -2)
    Hk = sp.diags(sk_power_2.ravel(), 0).todense()

    # Solve for the X which appears in the inequality for step size
    X = linalg.spsolve((A@Hk@A.T), b).reshape(constraints, 1)
    #print("Zero??", A.T@X)

    # Solve for tk_bar
    tk_bar = (np.divide(sk, A.T@X))

    #print("TK", tk_bar.size)

    # Find the minimum value of tk_bar from the tk_bar vector
    tk_bar_min = 3000000
    for index, element in enumerate(tk_bar):
      if (element > 0) & (element < tk_bar_min):
        tk_bar_min = element[0]
        #print(tk_bar_min)

    # Find a value slightly lower than tk_bar so that the value is less than and not equal to zero
    tk_min = 0.9*tk_bar_min

    # Calculate y
    ykplus1 = yk + tk_min*X

    # Calculate the value of the objective function
    obj_func = (b.T@ykplus1)[0][0]

    i_list.append(i)
    y_list.append(obj_func)

    # print the value of step increment and objective function
    #print("Obj function:", obj_func)
    #print("Step increment:", tk_min)

    # set yk as new y_k+1 for the next iteration
    yk = ykplus1


    if(y_list[-1] - y_list[-2] < 1):
      iteration_condition = 0
    #print(i_list)
    #print(y_list)
  plt.figure(1, figsize=(7,7))
  plt.ylabel('Objective Function')
  plt.xlabel('Iterations')
  plt.plot(i_list, y_list)
  return ykplus1


def get_sudoku_from_dual(y_star, A):
  c = np.ones(A.shape[1]).reshape(A.shape[1], 1)

  w_star = A.T@y_star + c

  w_star_bar = w_star.reshape(9, 9, 9)
  sudoku = np.argmax(w_star_bar, axis=2) + 1
  return sudoku

def calculate_kkt_soln(y_star, A):
  pass

if __name__ == '__main__':
    # 0 means unknown
    S_easy = np.array([[0,1,0,7,0,8,9,0,0],
                       [3,8,0,0,0,0,0,0,0],
                       [0,0,9,0,0,5,6,0,0],
                       [0,9,0,0,7,0,0,0,0],
                       [0,3,1,0,0,0,0,2,0],
                       [0,0,0,4,5,0,0,8,0],
                       [0,5,0,0,6,2,4,9,0],
                       [6,7,3,0,4,9,0,5,1],
                       [0,4,0,0,0,0,0,0,3]])

    S_hard = np.array([[0,0,3,0,0,9,0,8,1],
                       [0,0,0,2,0,0,0,6,0],
                       [5,0,0,0,1,0,7,0,0],
                       [8,9,0,0,0,0,0,0,0],
                       [0,0,5,6,0,1,2,0,0],
                       [0,0,0,0,0,0,0,3,7],
                       [0,0,9,0,2,0,0,0,8],
                       [0,7,0,0,0,4,0,0,0],
                       [2,5,0,8,0,0,6,0,0]])
    S_r_t = np.array([[0,2,3,5,7,9,0,8,1],
                       [9,1,7,2,4,8,3,6,0],
                       [5,8,0,3,1,6,7,9,2],
                       [8,9,2,4,3,0,0,5,6],
                       [7,3,5,6,8,1,2,0,9],
                       [1,4,6,9,5,2,8,3,7],
                       [3,0,9,0,2,5,0,0,8],
                       [0,7,8,1,0,4,9,2,0],
                       [2,5,1,8,9,0,6,0,3]])

    # demo visualization

    # demo code for visualizing sparse matrices
    plt.figure(figsize=(15, 20))
    plt.spy(get_Arow())
    plt.show()
    plt.figure(figsize=(15, 20))
    plt.spy(get_Acol())
    plt.show()
    plt.figure(figsize=(15, 20))
    plt.spy(get_Abox())
    plt.show()
    plt.figure(figsize=(15, 25))
    plt.spy(get_Acell())
    plt.show()
    plt.figure(figsize=(15, 25))
    plt.spy(get_Aclue(S_easy, 9), marker="s")
    #print("Shape Aeasy", get_Aclue(S_easy, 9).shape)
    plt.show()
    plt.figure(figsize=(15, 20))
    plt.spy(get_Aclue(S_hard, 9), marker="s")
    #print("Shape Ahard", get_Aclue(S_hard, 9).shape)
    #print(get_Aclue(S_hard, 9))
    plt.show()

    # Solve sudoku

    # EASY
    a_easy = get_A(S_easy, 9)
    print_sudoku(S_easy, S_easy, '\33[35m', '\33[30m')
    y_easy = dual_affine_scaling(a_easy)
    solution_easy = get_sudoku_from_dual(y_easy, a_easy)
    print("Easy sudoku")
    print_sudoku(solution_easy, S_easy, '\33[35m', '\33[30m')

    # HARD
    a_hard = get_A(S_hard, 9)
    y_hard = dual_affine_scaling(a_hard)
    print_sudoku(S_hard, S_hard, '\33[35m', '\33[30m')
    #print(a_hard.shape)
    solution_hard = get_sudoku_from_dual(y_hard, a_hard)
    print("Hard sudoku")
    print_sudoku(solution_hard, S_hard, '\33[35m', '\33[30m')

    # HARD ROUND TWO
    # a_round_two = get_A(S_r_t, 9)
    # y_round_two = dual_affine_scaling(a_round_two)
    # #print_sudoku(round_two, round_two, '\33[35m', '\33[30m')
    # print(a_round_two.shape)
    # solution_round_two = get_sudoku_from_dual(y_round_two, a_round_two)
    # print_sudoku(solution_round_two, S_r_t, '\33[35m', '\33[30m')
    # print(a_hard.shape)
