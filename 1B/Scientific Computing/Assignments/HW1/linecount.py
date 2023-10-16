import os
import sys
from mpi4py import MPI

def count_lines(filename):
    with open(filename, 'r') as file:
        return sum(1 for line in file)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) != 2:
        if rank == 0:
            print("Usage: mpirun -np <N> python linecount.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    if rank == 0:
        txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    else:
        txt_files = None

    txt_files = comm.bcast(txt_files, root=0)
    num_files = len(txt_files)

    files_per_process = num_files // size
    remainder = num_files % size

    start_idx = rank * files_per_process + min(rank, remainder)
    end_idx = start_idx + files_per_process + (1 if rank < remainder else 0)

    max_lines = 0
    local_max_lines = 0

    for idx in range(start_idx, end_idx):
        file_path = os.path.join(directory, txt_files[idx])
        line_count = count_lines(file_path)
        local_max_lines = max(local_max_lines, line_count)

    max_lines = comm.reduce(local_max_lines, op=MPI.MAX, root=0)

    if rank == 0:
        print(f"Maximum number of lines in {sys.argv[1]}/*.txt files: {max_lines}")

if __name__ == "__main__":
    main()
