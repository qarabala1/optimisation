import numpy as np

# Параметры задачи
n = 6  # размерность вектора x
lambda_ = 10**(-4)  # шаг метода градиентного спуска
tolerance = 10**(-6)  # условие остановки

# Генерация положительно определённой матрицы A
def generate_positive_definite_matrix(size):
    A = np.random.rand(size, size)
    return np.dot(A, A.T)

# Функция для проверки положительной определённости
def is_positive_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)

# Функция для вычисления f(x)
def f(x, A, b):
    return 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b.T, x)

# Градиент функции
def grad_f(A, x, b):
    return np.dot(A, x) + b

# Генерация данных
A = generate_positive_definite_matrix(n)
b = np.random.rand(n, 1)

# Проверка на положительную определённость
if is_positive_definite(A):
    print("Матрица A положительно определённая.")
else:
    print("Матрица A не положительно определённая.")

# Точное решение
x_exact = -np.linalg.inv(A).dot(b)

# Градиентный спуск
x_k = np.random.rand(n, 1)  # начальная точка
steps = 0
intermediate_results = []  # для хранения промежуточных результатов

# Выполнение градиентного спуска
while True:
    grad = grad_f(A, x_k, b)  # Векторизованное вычисление градиента
    x_next = x_k - lambda_ * grad
    steps += 1

    # Проверка условия остановки
    if np.linalg.norm(x_next - x_k) < tolerance:
        break

    x_k = x_next

    # Хранение только промежуточного результата через 1/4, 1/2, 3/4 шагов
    if steps % (steps // 4 + 1) == 0:
        intermediate_results.append((steps, x_k.copy(), f(x_k, A, b)))

# Финальные результаты
x_final = x_k
f_final = f(x_final, A, b)
f_exact = f(x_exact, A, b)

# Определяем шаги 1/4, 1/2 и 3/4
ksh_steps = len(intermediate_results)
ksh_1_4 = intermediate_results[int(ksh_steps / 4)] if ksh_steps >= 4 else None
ksh_1_2 = intermediate_results[int(ksh_steps / 2)] if ksh_steps >= 2 else None
ksh_3_4 = intermediate_results[int(3 * ksh_steps / 4)] if ksh_steps >= 3 else None

# Погрешности метода градиентного спуска
errors = {
    "xm1 - xточ1": x_final[0] - x_exact[0],
    "xm2 - xточ2": x_final[1] - x_exact[1],
    "xm3 - xточ3": x_final[2] - x_exact[2],
    "xm4 - xточ4": x_final[3] - x_exact[3],
    "xm5 - xточ5": x_final[4] - x_exact[4],
    "xm6 - xточ6": x_final[5] - x_exact[5],
    "fxm - fx*": f_final - f_exact
}

# Вывод результатов
print(f"Точное решение: {x_exact.flatten()}")
print(f"Финальное решение после {steps} шагов: {x_final.flatten()}")
print(f"Значение функции в точке x*: {f_exact}")
print(f"Значение функции в финальной точке: {f_final}")

if ksh_1_4:
    print(f"xКШ/4 = {ksh_1_4[1].flatten()}, fxКШ/4 = {ksh_1_4[2]}")
if ksh_1_2:
    print(f"xКШ/2 = {ksh_1_2[1].flatten()}, fxКШ/2 = {ksh_1_2[2]}")
if ksh_3_4:
    print(f"x3КШ/4 = {ksh_3_4[1].flatten()}, fx3КШ/4 = {ksh_3_4[2]}")

print("Погрешности метода градиентного спуска:")
for key, value in errors.items():
    print(f"{key}: {value.flatten()}")
