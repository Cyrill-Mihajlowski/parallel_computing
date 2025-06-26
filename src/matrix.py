import time
from multiprocessing import Pool

import numpy as np
import torch


class Matrix:
    def __init__(self):
        pass

    def matrixBase(self, limit: int = 1024):
        """Вычисление матриц без параллейной обработки

        Простое последовательное умножение

        Args:
            number (int, optional): Лимит матрицы. Defaults to 4096.
        """

        # Размер матриц
        N = limit

        # Генерация случайных матриц A и B
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)

        # Однопоточное умножение матриц
        start_time = time.time()
        C = np.zeros((N, N), dtype=np.float32)

        for i in range(N):
            for j in range(N):
                C[i, j] = sum(A[i, k] * B[k, j] for k in range(N))
            if i % 100 == 0:
                print(f"Обработано {i} строк из {N}")

        end_time = time.time()

        # Время выполнения
        execution_time = end_time - start_time

        # Количество операций (умножение + сложение)
        num_operations = 2 * (N**3)  # 2 операции на один элемент C[i, j]

        # Производительность (FLOPS)
        flops = num_operations / execution_time

        print(f"Время выполнения: {execution_time:.2f} секунд")
        print(f"Производительность: {flops / 1e9:.2f} GFLOPS")

    def multiply_row(self, args):
        """Вычисляет одну строку результирующей матрицы C."""
        A, B, row = args
        N = A.shape[1]
        C_row = np.zeros(N, dtype=np.float32)
        for j in range(N):
            C_row[j] = sum(A[row, k] * B[k, j] for k in range(N))
        return row, C_row

    def matrix_multiprocessing(self, limit: int = 4096, num_processes: int = 4):
        """Вычисление матриц с использованием многопроцессорности.

        Многопоточность и многопроцессорность
        """
        # Размер матриц
        N = limit

        # Генерация случайных матриц A и B
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)
        C = np.zeros((N, N), dtype=np.float32)

        # Многопроцессорное умножение матриц
        start_time = time.time()

        with Pool(processes=num_processes) as pool:
            results = pool.map(self.multiply_row, [(A, B, row) for row in range(N)])

        for row, C_row in results:
            C[row, :] = C_row

        end_time = time.time()

        # Время выполнения
        execution_time = end_time - start_time

        # Количество операций (умножение + сложение)
        num_operations = 2 * (N**3)

        # Производительность (FLOPS)
        flops = num_operations / execution_time

        print(f"Время выполнения (многопроцессорное): {execution_time:.2f} секунд")
        print(f"Производительность: {flops / 1e9:.2f} GFLOPS")

    def matrix_numpy_blas(self, limit: int = 4096):
        """BLAS (Basic Linear Algebra Subprograms)

        BLAS через NumPy
        """

        N = limit
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)

        start_time = time.time()
        C = np.dot(A, B)
        end_time = time.time()

        execution_time = end_time - start_time
        num_operations = 2 * (N**3)
        flops = num_operations / execution_time

        print(f"Время выполнения (NumPy BLAS): {execution_time:.2f} секунд")
        print(f"Производительность: {flops / 1e9:.2f} GFLOPS")

        return C  # Возвращаем результат для анализа

    def matrix_torch(self, limit: int = 4096):
        """Вычисления с использованием Cuda ядер."""
        N = limit

        # Генерация случайных матриц на GPU
        A = torch.rand((N, N), dtype=torch.float32, device="cuda")
        B = torch.rand((N, N), dtype=torch.float32, device="cuda")

        # Вычисление произведения
        start_time = time.time()
        C = torch.matmul(A, B)
        end_time = time.time()

        # Время выполнения
        execution_time = end_time - start_time

        # Подсчет FLOPS (2 операции на каждый элемент результирующей матрицы)
        num_operations = 2 * (N**3)  # Умножение + сложение
        flops = num_operations / execution_time

        print(f"Время выполнения (GPU): {execution_time:.2f} секунд")
        print(f"Производительность: {flops / 1e9:.2f} GFLOPS")

        return C
