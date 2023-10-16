import numpy as np
import sys
import time
import random
import matplotlib.pyplot as plt
from copy import copy, deepcopy

M = 1000
recommended_costs = np.zeros(M, dtype=int)
random_costs = np.zeros(M, dtype=int)

# Define a function to take input preferences for student-project assignments
def take_input():
    preferences = []  # Initialize an empty list to store preferences

    try:
        N = int(sys.argv[1])  # Try to read the first command-line argument as the number of students or projects
        preferences = np.asarray([np.random.choice(np.arange(0, N), replace=False, size=(N)) for _ in range(N)])
        # Generate a random preferences matrix based on the given number
        assert preferences.shape == (N, N), "Preferences matrix is wrong!"  # Check if the matrix is square
    except IndexError:
        print("Enter one argument (either a number to randomly generate preferences or a .txt file)")
        # If no command-line argument is provided, print an error message and exit
        exit(1)
    except ValueError:
        file = sys.argv[1]  # Read the first command-line argument as a file name
        with open(file, "r") as f:
            # Open the file for reading
            preferences = np.asarray([[int(i) for i in j.replace("\n", "").split(",")] for j in f.readlines()])
            # Read preferences from the file and convert them to a NumPy array
        
        assert preferences.shape[0] == preferences.shape[1], "Preferences matrix must be square!"
        # Check if the matrix read from the file is square

    return preferences.shape[0], preferences  # Return the size (N) and the preferences matrix


def mark_matrix(mat):
    current_matrix = mat  # Make a copy of the input matrix
    zero_bool_mat = (current_matrix == 0)  # Create a boolean matrix to identify zeros
    zero_bool_mat_copy = zero_bool_mat.copy()  # Create a copy of the boolean matrix

    marked_zero = []  # List to store coordinates of marked zeros
    while (True in zero_bool_mat_copy):  # Continue until there are unmarked zeros
        min_row = [99999, -1]  # Initialize variables to track the minimum row

        for row_num in range(zero_bool_mat_copy.shape[0]):
            if np.sum(zero_bool_mat_copy[row_num] == True) > 0 and min_row[0] > np.sum(zero_bool_mat_copy[row_num] == True):
                # Find the row with the fewest unmarked zeros
                min_row = [np.sum(zero_bool_mat_copy[row_num] == True), row_num]

        zero_index = np.where(zero_bool_mat_copy[min_row[1]] == True)[0][0]  # Find the first unmarked zero in the row
        marked_zero.append((min_row[1], zero_index))  # Mark the row and column of the zero
        zero_bool_mat_copy[min_row[1], :] = False  # Mark the row as processed
        zero_bool_mat_copy[:, zero_index] = False  # Mark the column as processed

    marked_zero_row = []  # List to store row indexes with marked zeros
    marked_zero_col = []  # List to store column indexes with marked zeros
    for i in range(len(marked_zero)):
        marked_zero_row.append(marked_zero[i][0])
        marked_zero_col.append(marked_zero[i][1])

    non_marked_row = list(set(range(current_matrix.shape[0])) - set(marked_zero_row))
    # Find rows without marked zeros

    marked_cols = []  # List to store column indexes with marked zeros
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

    marked_rows = list(set(range(mat.shape[0])) - set(non_marked_row))
    # Find rows with marked zeros

    return (marked_zero, marked_rows, marked_cols)  # Return marked zeros, marked rows, and marked columns


def step(mat):
    for row_num in range(mat.shape[0]):
        mat[row_num] = mat[row_num] - np.min(mat[row_num])
    # Subtract the minimum value of each row from the entire row
    
    for col_num in range(mat.shape[1]):
        mat[:, col_num] = mat[:, col_num] - np.min(mat[:, col_num])
    # Subtract the minimum value of each column from the entire column

    return mat  # Return the matrix after performing the step


def set_matrix(mat, cover_rows, cover_cols):
    current_matrix = mat
    non_zero_element = []
    
    # Collect non-zero elements not covered by marked rows or columns
    for row in range(len(current_matrix)):
        if row not in cover_rows:
            for i in range(len(current_matrix[row])):
                if i not in cover_cols:
                    non_zero_element.append(current_matrix[row][i])
    
    min_num = min(non_zero_element)

    # Subtract the minimum non-zero element from uncovered elements in rows
    for row in range(len(current_matrix)):
        if row not in cover_rows:
            for i in range(len(current_matrix[row])):
                if i not in cover_cols:
                    current_matrix[row, i] = current_matrix[row, i] - min_num

    # Add the minimum non-zero element to covered elements in rows and columns
    for row in range(len(cover_rows)):  
        for col in range(len(cover_cols)):
            current_matrix[cover_rows[row], cover_cols[col]] = current_matrix[cover_rows[row], cover_cols[col]] + min_num

    return current_matrix

def student_project_distribution(cost_matrix):
    n = cost_matrix.shape[0]
    current_matrix = deepcopy(cost_matrix)
    
    current_matrix = step(current_matrix)
    
    count_zero_lines = 0

    # Continue marking rows and columns until all rows or columns have zeros   
    while count_zero_lines < n:
        ans_pos, marked_rows, marked_cols = mark_matrix(current_matrix)
        count_zero_lines = len(marked_rows) + len(marked_cols)

        # Set the matrix to cover marked rows and columns
        if count_zero_lines < n:
            current_matrix = set_matrix(current_matrix, marked_rows, marked_cols)
    
        return ans_pos
    

def plot(*args):
    plt.hist(recommended_costs, bins=100, alpha=0.5, label='Recommended')
    plt.hist(random_costs, bins=100, alpha=0.5, label='Random')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    

N, preferences = take_input()

start_all = time.time()

for i in range(M):
    if i == 0: start = time.time()
    distribution = student_project_distribution(preferences)
    if i == 0: 
        end = time.time() 
        print(preferences)
    
    total = 0
    for val in distribution:
        if i == 0: print(f"Student {val[0]+1} is assigned to Project #{val[1]+1}.")
        total += preferences[val[0], val[1]]

    recommended_costs[i] = total

    for j in range(N):
        np.random.shuffle(preferences[j])

    if i == 0: print(f"The total cost of the distribution is {total}")
    if i == 0: print(f"Time elapsed for 1 iteration: {end-start:.8f} s.")

    tmp_preferences = copy(preferences)

    for j in range(N):
        col = random.randint(0, tmp_preferences.shape[1]-1)
        random_costs[i] += tmp_preferences[j][col]
        tmp_preferences = np.delete(tmp_preferences, col, 1)

print(f"Total time elapsed: {time.time() - start_all} s.")

mean_rec = recommended_costs.mean()
mean_ran = random_costs.mean()

print(f"Mean recommended: {mean_rec}")
print(f"Mean random: {mean_ran}")

plot(recommended_costs, random_costs)