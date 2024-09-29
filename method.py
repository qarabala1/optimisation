import numpy as np

# Проверка положительной определенности матрицы A
def is_positive_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)

# Генерация положительно определённой матрицы A
def generate_positive_definite_matrix(size):
    while True:
        A = np.random.rand(size, size)
        if is_positive_definite(A):
            return A

# Размер матрицы
n = 6

# Генерация положительно определённой матрицы A (6x6)
A = generate_positive_definite_matrix(n)

# Генерация произвольного ненулевого вектора b (6x1)
b = np.random.rand(n, 1)

# Генерация начального вектора x0 (6x1), отдаленного от решения
x0 = np.random.rand(n, 1)

# Решение системы x* = -A^-1 * b
x_star = -np.linalg.inv(A).dot(b)



