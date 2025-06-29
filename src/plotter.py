import matplotlib.pyplot as plt


def plot_performance(
    cpu_sizes, accel_sizes, single_core_times, multi_core_times, blas_times, cuda_times
):
    """
    Строит два графика:
    1. Сравнение одноядерного и многоядерного методов (CPU).
    2. Сравнение методов BLAS и CUDA (ускоренные методы).
    """
    # Построение графика для одноядерного и многоядерного методов
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(
        cpu_sizes, single_core_times, label="Single Core", marker="o", color="blue"
    )
    plt.plot(cpu_sizes, multi_core_times, label="Multi Core", marker="o", color="green")
    plt.title("Single vs Multi Core Performance")
    plt.xlabel("Matrix Size")
    plt.ylabel("Execution Time (s)")
    plt.legend()
    plt.grid(True)

    # Построение графика для BLAS и CUDA методов
    plt.subplot(1, 2, 2)
    plt.plot(accel_sizes, blas_times, label="BLAS", marker="o", color="orange")
    if any(cuda_times):  # Убедиться, что CUDA данные существуют
        plt.plot(accel_sizes, cuda_times, label="CUDA", marker="o", color="red")
    plt.title("BLAS vs CUDA Performance")
    plt.xlabel("Matrix Size")
    plt.ylabel("Execution Time (s)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
