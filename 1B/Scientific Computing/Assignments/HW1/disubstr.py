import numpy as np

def invPerm(p):
    '''Invert the permutation p'''
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


def build_suffix_array(A):
    if not type(A) is np.ndarray:
        A = np.array(list(A))
    N = len(A)
    M = int(np.ceil(np.log2(N)))+1   # number of iterations

    # auxiliary arrays; row m stores results after m'th step:

    # positions of sorted length-(2**m) sequences in A
    P = np.zeros((M,N), dtype=int)

    # rank (0, 1, etc.) of sorted length-(2**m) sequences after sorting
    Q = np.zeros((M,N), dtype=int)

    # rank of sorted length-(2**m) sequences at its starting position in A;
    # padded by 0 on the right
    R = np.zeros((M,3*N), dtype=int)

    for k in range(M):
        if k == 0:
            P[0] = np.argsort(A)
            Q[0][1:] = np.cumsum(A[P[0]][1:] != A[P[0]][:-1])
            R[0][:N] = Q[0][invPerm(P[0])]
        else:
            offset = 2**(k-1)
            r = np.lexsort((R[k-1, P[k-1]+offset], R[k-1, P[k-1]]))
            P[k] = P[k-1][r]
            # k'th rank increases iff (k-1)'th rank increases at least for one element of the pair
            Q[k][1:] = np.cumsum(np.logical_or(R[k-1][P[k]][1:] != R[k-1][P[k]][:-1],
                                          R[k-1][P[k]+offset][1:] != R[k-1][P[k]+offset][:-1]))
            R[k][:N] = Q[k][invPerm(P[k])]

            # early stopping if suffixes already fully sorted (max rank is N-1)
            if Q[k][-1] == N-1:
                break

    SA = P[k]
    return SA, P[:k+1], Q[:k+1], R[:k+1]

def getLCP(SA, R):
    (M, N) = R.shape
    LCP = np.zeros((len(SA)-1,),dtype=int)
    for m in range(M-1)[::-1]:
        t = (R[m][SA[1:]+LCP] == R[m][SA[:-1]+LCP]).astype(int)
        LCP += (2**m)*t
    return LCP

def count_distinct_substrings(s):
    SA, _, _, _ = build_suffix_array(s)
    n = len(s)

    # Initialize rank and lcp (longest common prefix) arrays
    LCP = [0] * n
    R = [0] * n

    k = 0  # Initialize the length of the common prefix
    distinct_substrings = 0

    # Calculate rank array
    for i in range(n):
        R[SA[i]] = i

    for i in range(n):
        if R[i] == n - 1:
            k = 0
            continue

        j = SA[R[i] + 1]

        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1

        LCP[R[i]] = k

        if k > 0:
            k -= 1
    
    # Calculate the number of distinct substrings
    distinct_substrings = n - SA[0]
    for i in range(1, n):
        distinct_substrings += (n - SA[i] - LCP[i - 1])

    return distinct_substrings

T = int(input())
L = []
for _ in range(T):
    test_case = input()
    L.append(test_case)

for i in L:
    result = count_distinct_substrings(i)
    print(result)
