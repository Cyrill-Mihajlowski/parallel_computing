import numpy as np
import torch

from matrix import Matrix
from plotter import plot_performance


def main():
    """
    Главная функция программы, выполняющая тесты производительности
    различных методов умножения матриц.

    Тесты проводятся для следующих категорий:
    1. Однопоточное вычисление на CPU.
    2. Многопоточное вычисление на CPU.
    3. BLAS через NumPy.
    4. Вычисление на GPU через PyTorch (CUDA).

    Генерируются графики для визуализации производительности методов.
    """

    mat = Matrix()

    # Размеры матриц для каждой категории
    cpu_sizes = [128, 256, 512, 1024]
    accel_sizes = [4096, 8192, 16384, 32768]

    # Списки для хранения времени выполнения
    single_core_times = []
    multi_core_times = []
    blas_times = []
    cuda_times = []

    # Тесты для CPU (одноядерный и многоядерный)
    for size in cpu_sizes:
        print(f"CPU тесты для размера матрицы: {size}x{size}")

        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)

        # Однопоточная обработка
        single_core_time = mat.matrix_base(A, B)
        single_core_times.append(single_core_time)

        # Многопоточная обработка
        multi_core_time = mat.matrix_multiprocessing(A, B, num_processes=8)
        multi_core_times.append(multi_core_time)

    # Тесты для ускоренных методов (BLAS и CUDA)
    for size in accel_sizes:
        print(f"Ускоренные тесты для размера матрицы: {size}x{size}")

        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)

        # BLAS через NumPy
        blas_time = mat.matrix_numpy_blas(A, B)
        blas_times.append(blas_time)

        # CUDA через PyTorch (если доступно)
        if torch.cuda.is_available():
            A_cuda = torch.tensor(A, device="cuda")
            B_cuda = torch.tensor(B, device="cuda")
            cuda_time = mat.matrix_torch(A_cuda, B_cuda)
            cuda_times.append(cuda_time)
        else:
            cuda_times.append(None)

    # Генерация графиков

    plot_performance(
        cpu_sizes,
        accel_sizes,
        single_core_times,
        multi_core_times,
        blas_times,
        cuda_times,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nЗавершение работы.")
