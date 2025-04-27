import numpy as np
import matplotlib.pyplot as plt
import time

def samuelson_berkowitz(A):
    n = A.shape[0]
    coefficients = [1]  # Начинаем с c_0 = 1

    for k in range(1, n + 1):
        sub_A = A[:k, :k]  # Подматрица размера k x k
        trace_term = np.trace(sub_A)
        if k == 1:
            coefficients.append(-trace_term)
        else:
            new_coeff = -trace_term + sum(
                coefficients[j] * np.trace(np.linalg.matrix_power(sub_A, k - j))
                for j in range(1, k)
            )
            coefficients.append(new_coeff)

    return np.array(coefficients)


def compute_metrics():
    sizes = list(range(2, 151))
    times = []
    errors = []

    for n in sizes:
        A = np.random.rand(n, n)

        start_time = time.time()
        coeffs = samuelson_berkowitz(A)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)

        exact_eigenvalues = np.linalg.eigvals(A)
        approx_eigenvalues = np.roots(coeffs[::-1])

        error = np.mean(np.abs(np.sort(exact_eigenvalues) - np.sort(approx_eigenvalues)))
        errors.append(error)

    return sizes, times, errors

sizes, times, errors = compute_metrics()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(sizes, times, label="Количество операций")
plt.xlabel("Размер матрицы")
plt.ylabel("Время (с)")
plt.title("Зависимость количества операций от размерности матрицы")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(sizes, errors, label="Средняя погрешность", color="red")
plt.xlabel("Размер матрицы")
plt.ylabel("Средняя ошибка")
plt.title("Зависимость погрешности от размерности матрицы")
plt.legend()

plt.tight_layout()
plt.show()
