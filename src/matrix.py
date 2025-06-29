import time
from multiprocessing import Pool

import numpy as np
import torch


class Matrix:
    def matrix_base(self, A, B):
        """Вычисление матриц без параллельной обработки."""
        N = A.shape[0]
        C = np.zeros((N, N), dtype=np.float32)

        start_time = time.time()
        for i in range(N):
            for j in range(N):
                C[i, j] = sum(A[i, k] * B[k, j] for k in range(N))
        execution_time = time.time() - start_time
        return execution_time

    def multiply_row(self, args):
        A, B, row = args
        N = A.shape[1]
        C_row = np.zeros(N, dtype=np.float32)
        for j in range(N):
            C_row[j] = sum(A[row, k] * B[k, j] for k in range(N))
        return row, C_row

    def matrix_multiprocessing(self, A, B, num_processes):
        """Вычисление матриц с использованием многопроцессорности."""
        N = A.shape[0]
        C = np.zeros((N, N), dtype=np.float32)

        start_time = time.time()
        with Pool(processes=num_processes) as pool:
            results = pool.map(self.multiply_row, [(A, B, row) for row in range(N)])

        for row, C_row in results:
            C[row, :] = C_row
        execution_time = time.time() - start_time
        return execution_time

    def matrix_numpy_blas(self, A, B):
        """BLAS через NumPy."""
        start_time = time.time()
        C = np.dot(A, B)
        execution_time = time.time() - start_time
        return execution_time

    def matrix_torch(self, A, B):
        """Вычисления с использованием CUDA ядер."""
        start_time = time.time()
        C = torch.matmul(A, B)
        execution_time = time.time() - start_time
        return execution_time
