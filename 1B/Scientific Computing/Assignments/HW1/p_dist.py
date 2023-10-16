#Problem 4
import time
import numpy as np
import copy
from scipy.optimize import linear_sum_assignment
N = int(input("Enter the number of projects and students: "))

# Create an empty matrix
preferences_matrix = np.zeros((N, N), dtype=int)

# Generate a list of values from 0 to N-1
values = list(range(N))

# Shuffle the values and assign them to each row
for i in range(N):
    np.random.shuffle(values)
    preferences_matrix[i] = values.copy()  # Use a copy to keep the original order

print(preferences_matrix)
def hungarian_step (mat): 
    for row_num in range(mat.shape[0]): 
        mat[row_num] = mat[row_num] - np.min(mat[row_num])

    for col_num in range(mat.shape[1]): 
        mat[:,col_num] = mat[:,col_num] - np.min(mat[:,col_num])
    
    return mat
def min_zeros(zero_mat, mark_zero):
    min_row = [99999, -1] #to find the row with minimal zeroes

    for row_num in range(zero_mat.shape[0]): 
        if np.sum(zero_mat[row_num] == True) > 0 and min_row[0] > np.sum(zero_mat[row_num] == True):
            min_row = [np.sum(zero_mat[row_num] == True), row_num]

    # Marked the specific row and column as False
    zero_index = np.where(zero_mat[min_row[1]] == True)[0][0]
    mark_zero.append((min_row[1], zero_index))
    zero_mat[min_row[1], :] = False
    zero_mat[:, zero_index] = False
def mark_matrix(mat):
    #Transform the matrix to boolean matrix(0 = True, others = False)
    cur_mat = mat
    zero_bool_mat = (cur_mat == 0)
    zero_bool_mat_copy = zero_bool_mat.copy()

    #Recording possible answer positions by marked_zero
    marked_zero = []
    while (True in zero_bool_mat_copy):
        min_zeros(zero_bool_mat_copy, marked_zero)

    #Recording the row and column indexes seperately.
    marked_zero_row = []
    marked_zero_col = []
    for i in range(len(marked_zero)):
        marked_zero_row.append(marked_zero[i][0])
        marked_zero_col.append(marked_zero[i][1])
    
    # mark rows not containing zeros
    non_marked_row = list(set(range(cur_mat.shape[0])) - set(marked_zero_row))
    
    # mark columns with zeros
    marked_cols = []
    check_switch = True
    while check_switch:
        check_switch = False
        for i in range(len(non_marked_row)):
            row_array = zero_bool_mat[non_marked_row[i], :]
            for j in range(row_array.shape[0]):
                if row_array[j] == True and j not in marked_cols:

                    marked_cols.append(j)
                    check_switch = True

        for row_num, col_num in marked_zero:
            if row_num not in non_marked_row and col_num in marked_cols:
                
                non_marked_row.append(row_num)
                check_switch = True
    
    # mark rows with zeros
    marked_rows = list(set(range(mat.shape[0])) - set(non_marked_row))
    
    return(marked_zero, marked_rows, marked_cols)
def adjust_matrix(mat, cover_rows, cover_cols):
    cur_mat = mat
    non_zero_element = []
    
    # find the minimum value of an element not in a marked column/row 
    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    non_zero_element.append(cur_mat[row][i])
    
    min_num = min(non_zero_element)

    # substract to all values not in a marked row/column
    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    cur_mat[row, i] = cur_mat[row, i] - min_num
    # add to all values in marked rows/column
    for row in range(len(cover_rows)):  
        for col in range(len(cover_cols)):
            cur_mat[cover_rows[row], cover_cols[col]] = cur_mat[cover_rows[row], cover_cols[col]] + min_num

    return cur_mat

t0 = time.time()
def hungarian_algorithm(cost_matrix):
    n = cost_matrix.shape[0]
    cur_mat = copy.deepcopy(cost_matrix)
    
    cur_mat = hungarian_step(cur_mat)
    
    count_zero_lines = 0
        
    while count_zero_lines < n:
        ans_pos, marked_rows, marked_cols = mark_matrix(cur_mat)
        count_zero_lines = len(marked_rows) + len(marked_cols)

        if count_zero_lines < n:
            cur_mat = adjust_matrix(cur_mat, marked_rows, marked_cols)
    
        return ans_pos
assignment = hungarian_algorithm(preferences_matrix)
print(f"The final assignment is: {assignment}")
print(preferences_matrix)
f = 0
for i in range(len(assignment)):
 f+=preferences_matrix[assignment[i][0]][assignment[i][1]]
 print(f"Student {i}: ", preferences_matrix[assignment[i][0]][assignment[i][1]])
print("objective function 𝑓 = ", f)
dt = time.time()-t0
print(dt)
#Experiment for problem 4
import matplotlib.pyplot as plt
import random
# Number of experiments
M = 1000
N = int(input("Enter the number of projects and students: "))
f_recommended = np.zeros(M, dtype=int)
f_random = np.zeros(M, dtype=int)
for i in range(M):
 
 preferences_matrix = np.zeros((N, N), dtype=int)
 values = list(range(N))
 for j in range(N):
  np.random.shuffle(values)
  preferences_matrix[j] = values.copy()  
 
 #recommended distribution
 assignment = hungarian_algorithm(preferences_matrix)
 for j in range(len(assignment)):
  f_recommended[i]+=preferences_matrix[assignment[j][0]][assignment[j][1]]
 
 #random distribution
 for j in range(N):
  colum_idx = random.randint(0,preferences_matrix.shape[1]-1)
  f_random[i]+= preferences_matrix[j][colum_idx]
  preferences_matrix = np.delete(preferences_matrix, colum_idx, 1)
 
mean_f_recommended = np.mean(f_recommended)
mean_f_random = np.mean(f_random)

# Plot the distributions
plt.hist(f_recommended, bins=40, alpha=0.5, label='Recommended algorithm')
plt.hist(f_random, bins=40, alpha=0.5, label='Random algorithm')
plt.xlabel('𝑓 Values')
plt.ylabel('Frequency')
plt.legend()
