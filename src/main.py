from matrix import Matrix


def main():
    mat = Matrix()

    # mat.matrixBase(limit=1024)  # Вычисление матриц без параллейной обработки
    # mat.matrix_multiprocessing(limit=2048, num_processes=8)  # Вычисление матриц с использованием многопроцессорности  # noqa: E501
    # mat.matrix_numpy_blas(limit=16384) # NumPy BLAS
    mat.matrix_torch(limit=32768)  # Вычислиния с импользованием Cuda ядер


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Завершение работы.")
