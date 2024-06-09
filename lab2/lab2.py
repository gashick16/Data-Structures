import numpy as np
from scipy.linalg import blas as scipy_blas
import time
from joblib import Parallel, delayed

n = 2048

# Генерация случайных матриц типа double
A = np.random.rand(n, n).astype(np.complex64)
B = np.random.rand(n, n).astype(np.complex64)

# оценка производитльности
def calculate_performance(time_taken, n):
    if time_taken == 0:
        return float('inf')
    c = 2 * n ** 3
    performance = c / time_taken * 1e-6  # в MFlops
    return performance

# алгебраическое перемножение матриц
def direct_matrix_multiply():
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.complex64)

    start_time = time.perf_counter()

    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

    end_time = time.perf_counter()

    # Время выполнения
    time_direct = end_time - start_time
    print(f"Время выполнения первого метода: {time_direct} seconds")

    # Считаем производительность
    performance_direct = calculate_performance(time_direct, n)
    print(f"Производительность первого метода: {performance_direct} MFlops")

    return C

# перемножение с использованием библотеки BLAS
def blas_matrix_multiply():
    start_time = time.perf_counter()

    C_blas = scipy_blas.cgemm(1.0, A, B)

    end_time = time.perf_counter()

    # Время выполнения
    time_blas = end_time - start_time
    print(f"Время выполнения второго метода: {time_blas} seconds")

    # Считаем производительность
    performance_blas = calculate_performance(time_blas, n)
    print(f"Производительность первого метода: {performance_blas} MFlops")

    return C_blas

# оптимизированное перемножение с использованием параллельных вычислений на CPU
def parallel_matrix_multiply(num_jobs=-1):
    def multiply_row(i):
        return np.dot(A[i, :], B)

    start_time = time.perf_counter()

    C = Parallel(n_jobs=num_jobs)(delayed(multiply_row)(i) for i in range(A.shape[0]))
    C_optimized = np.array(C)

    end_time = time.perf_counter()
    
    # Время выполнения
    time_optimized = end_time - start_time
    print(f"Время выполнения третьего метода: {time_optimized} seconds")

    # Считаем производительность
    performance_optimized = calculate_performance(time_optimized, n)
    print(f"Производительность третьего метода: {performance_optimized} MFlops")

    return C_optimized

# перемножение с использованием библотеки numpy
def numpy_matrix_multiply():
    start_time = time.perf_counter()

    C_numpy = np.dot(A, B)

    end_time = time.perf_counter()

    # Время выполнения
    time_numpy = end_time - start_time
    print(f"Время выполнения четвертого метода: {time_numpy} seconds")

    # Считаем производительность
    performance_numpy = calculate_performance(time_numpy, n)
    print(f"Производительность четвертого метода: {performance_numpy} MFlops")

    return C_numpy

# Сравнение матриц на совпадение
def compare_matrix(A, B, epsilon=1e-6):
    if np.allclose(A, B, atol=epsilon):
        print("Матрицы равны")
    else:
        print("Матрицы не равны")

# print()
# C_direct = direct_matrix_multiply()
print()
C_blas = blas_matrix_multiply()
print()
C_parallel = parallel_matrix_multiply()
print()
C_numpy = numpy_matrix_multiply()

# print("Сравнение матрицы Direct и BLAS")
# compare_matrix(C_direct, C_blas)
# print("Сравнение матрицы Direct и Parallel")
# compare_matrix(C_direct, C_parallel)
# print("Сравнение матрицы Direct и Numpy")
# compare_matrix(C_direct, C_numpy)