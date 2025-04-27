import numpy as np

EPSILON = 1e-6

def objective_function(A, x, b):
    return 0.5 * x.T @ A @ x + b @ x

def is_valid_matrix(A):
    def is_symmetric(a, tolerance=1e-8):
        return np.allclose(a, a.T, atol=tolerance)

    def is_non_singular(a):
        return np.linalg.det(a) != 0

    return is_symmetric(A) and is_non_singular(A)

def compute_jacobian(A: np.ndarray, x0: np.ndarray, x: np.ndarray):
    n = len(x0)
    J = np.zeros((n + 1, n + 1))
    J[:n, :n] = A + 2 * x[-1] * np.eye(n)
    J[:n, n] = 2 * (x[:n] - x0)
    J[n, :n] = 2 * (x[:n] - x0)
    return J

def residual_function(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    x: np.ndarray,
    radius: int,
) -> np.ndarray:
    return np.append(
        A @ x[:len(x0)] + 2 * x[-1] * (x[:len(x0)] - x0) + b,
        np.linalg.norm(x[:len(x0)] - x0) ** 2 - radius**2,
    )

def newton_method(A, x0, radius, b, initial_guess, epsilon=EPSILON):
    x_history = [initial_guess]

    f_val = residual_function(A, b, x0, initial_guess, radius)
    jacobian_matrix = compute_jacobian(A, x0, initial_guess)
    inv_jacobian = np.linalg.inv(jacobian_matrix)
    x_history.append(x_history[0] - inv_jacobian @ f_val)

    while np.linalg.norm(x_history[-1] - x_history[-2]) > epsilon:
        f_current = residual_function(A, b, x0, x_history[-1], radius)
        jacobian_current = compute_jacobian(A, x0, x_history[-1])
        inv_jacobian_current = np.linalg.inv(jacobian_current)
        x_history.append(x_history[-1] - inv_jacobian_current @ f_current)

    return x_history[-1]

A = np.array([
    [14.12034810510239, -21.14561905234863, 328.526781376923, 511.23841079056418],
    [-21.14561905234863, 229.192843485821, 355.0397007546424, 432.42160792754313],
    [328.526781376923, 355.0397007546424, -801.1289346824821, -623.7668985778583],
    [511.23841079056418, 432.42160792754313, -623.7668985778583, 133.14512451563],
])
b = np.array([4, 2, 1, 5])
x0 = np.array([1, 1, 1, 1])
radius = 4

if not is_valid_matrix(A):
    raise ValueError("Matrix A is not symmetric or singular")

# y = 0
x_min_solution = -np.linalg.inv(A) @ b

# y > 0
initial_guesses = [
    np.array([0.1, 0.2, 0.3, 0.4, radius]),
    np.array([0.6, 0.7, 0.8, 0.9, radius]),
    np.array([1.1, 1.2, 1.3, 1.4, radius]),
    np.array([1.6, 1.7, 1.8, 1.9, radius]),
    np.array([2.1, 2.2, 2.3, 2.4, radius]),
    np.array([2.6, 2.7, 2.8, 2.9, radius]),
    np.array([3.1, 3.2, 3.3, 3.4, radius]),
    np.array([4., 5., 6., 7., radius]),
]

results = [newton_method(A=A, x0=x0, radius=radius, b=b, initial_guess=start, epsilon=EPSILON) for start in initial_guesses]

# Output results
print(f"""
Проверка на невырожденность: {is_valid_matrix(A)}

Если y = 0: 
x* = {x_min_solution}
f = {objective_function(A, x_min_solution, b)}
норма = {np.linalg.norm(x_min_solution - x0)}
радиус = {radius}
норма <= радиус? {np.linalg.norm(x_min_solution - x0) <= radius}

Если  y > 0:
""")

for idx in range(len(results)):
    print(f"{idx + 1} f: {objective_function(A, results[idx][:4], b)} \t x: {results[idx]} \t y: {round(results[idx][-1], 5)} \n")
