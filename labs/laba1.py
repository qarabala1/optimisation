import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
n = 6  # Размерность вектора x
lambda_ = 1e-4  # Шаг метода градиентного спуска
tolerance = 1e-6  # Условие остановки

# Генерация положительно определённой матрицы A
def generate_positive_definite_matrix(size):
    A = np.random.rand(size, size)
    return np.dot(A, A.T)

# Функция для проверки положительной определённости
def is_positive_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)

# Функция для вычисления значения f(x)
def f(x, A, b):
    return 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b.T, x)

# Градиент функции f(x)
def grad_f(A, x, b):
    return np.dot(A, x) + b

# Генерация данных
A = generate_positive_definite_matrix(n)
b = np.random.rand(n, 1)

# Проверка на положительную определённость матрицы A
if is_positive_definite(A):
    print("Матрица A положительно определённая.")
else:
    print("Матрица A не положительно определённая.")

# Точное решение (x_exact)
x_exact = -np.linalg.inv(A).dot(b)

# Инициализация градиентного спуска
x_k = np.random.rand(n, 1)  # Начальная точка
steps = 0
intermediate_results = []  # Для хранения промежуточных результатов
values = []  # Для хранения значений функции на каждом шаге

# Вывод входных данных
print(f"Матрица A:\n{A}")
print(f"Вектор b:\n{b}")
print(f"Начальная точка x₀:\n{x_k}")

# Выполнение градиентного спуска
while True:
    grad = grad_f(A, x_k, b)  # Вычисление градиента
    x_next = x_k - lambda_ * grad
    steps += 1

    # Сохранение значения функции на каждом шаге
    values.append(f(x_k, A, b)[0][0])

    # Проверка условия остановки
    if np.linalg.norm(x_next - x_k) < tolerance:
        break

    # Обновление точки и сохранение промежуточных результатов
    x_k = x_next
    intermediate_results.append((steps, x_k.copy(), f(x_k, A, b)))

# Финальные результаты
x_final = x_k
f_final = f(x_final, A, b)
f_exact = f(x_exact, A, b)

# Определение промежуточных шагов 1/4, 1/2 и 3/4 от общего количества шагов
ksh_steps = steps
ksh_1_4 = intermediate_results[int(ksh_steps / 4)]     
ksh_1_2 = intermediate_results[int(ksh_steps / 2)] 
ksh_3_4 = intermediate_results[int(3 * ksh_steps / 4)] 

# Погрешности метода градиентного спуска
errors = {
    f"xm{i+1} - xточ{i+1}": x_final[i] - x_exact[i]
    for i in range(n)
}
errors["fxm - fx*"] = f_final - f_exact

# Вывод финальных результатов
print(f"\nТочное решение: {x_exact.flatten()}")
print(f"Финальное решение после {steps} шагов: {x_final.flatten()}")
print(f"Значение функции в точке x*: {f_exact}")
print(f"Значение функции в финальной точке: {f_final}")

# Вывод промежуточных шагов
if ksh_1_4:
    print(f"\nПромежуточный результат на 1/4 шагов: x = {ksh_1_4[1].flatten()}, f(x) = {ksh_1_4[2]}")
if ksh_1_2:
    print(f"Промежуточный результат на 1/2 шагов: x = {ksh_1_2[1].flatten()}, f(x) = {ksh_1_2[2]}")
if ksh_3_4:
    print(f"Промежуточный результат на 3/4 шагов: x = {ksh_3_4[1].flatten()}, f(x) = {ksh_3_4[2]}")

# Вывод погрешностей
print("\nПогрешности метода градиентного спуска:")
for key, value in errors.items():
    print(f"{key}: {value.flatten()}")

# Построение графика зависимости f(x) от номера шага
plt.plot(range(steps), values)
plt.xlabel('Номер шага')
plt.ylabel('Значение функции f(x)')
plt.title('Зависимость значения функции f(x) от номера шага')
plt.grid(True)
plt.show()
