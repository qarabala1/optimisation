import sys

import numpy as np
from tabulate import tabulate


ROUND_VAL = 3
sys.stdout = open("./labs/output.txt", "w", encoding="utf-8")

def print_matrix(matrix: np.ndarray, header: str = None):
    if len(matrix.shape) == 1:
        matrix = matrix.reshape((1, matrix.shape[0]))

    matrix = matrix.round(ROUND_VAL)
    str_matrix = [[str(cell) for cell in row] for row in matrix]
    s = f"{tabulate(str_matrix, tablefmt='fancy_grid' ,)}\n"

    if header is not None:
        header = str(header).center(len(s.split("\n")[0]))
        print(header)

    print(s)


def print_matrix_latex(matrix: np.ndarray, header: str = None):
    if len(matrix.shape) == 1:
        matrix = matrix.reshape((1, matrix.shape[0]))

    matrix = matrix.round(ROUND_VAL)
    str_matrix = [[str(cell) for cell in row] for row in matrix]
    s = (
        "$$ \\begin{bmatrix} "
        + " \\\\ ".join([" & ".join(row) for row in str_matrix])
        + " \\end{bmatrix} $$"
    )
    print(s)

def make_matrix(A: np.ndarray, b: np.ndarray, c: np.ndarray):
    return np.vstack(
        (
            np.hstack((np.reshape(b, (A.shape[0], 1)), A, np.eye(A.shape[0]))),
            np.hstack(((np.array([0])), c, np.zeros((A.shape[0])))),
        )
    )


def make_dual_matrix(A: np.ndarray, b: np.ndarray, c: np.ndarray):
    return np.vstack(
        (
            np.hstack((np.reshape(c, (A.shape[0], 1)), -A, np.eye(A.shape[0]))),
            np.hstack(((np.array([0])), -b, np.zeros((A.shape[0])))),
        )
    )


def simplex(simplex_matrix: np.ndarray):
    while True:
        index_of_element = simplex_matrix[-1, 1:].argmin()

        if simplex_matrix[-1, 1:][index_of_element] >= 0:
            break

        else:
            min_element = np.inf
            min_line = 0
            index_of_element += 1

            for line in range(simplex_matrix.shape[0] - 1):
                if (
                    simplex_matrix[line, index_of_element] > 0
                    and simplex_matrix[line, 0]
                    / simplex_matrix[
                        line,
                        index_of_element,
                    ]
                    < min_element
                ):
                    min_line = line
                    min_element = (
                        simplex_matrix[line, 0]
                        / simplex_matrix[
                            line,
                            index_of_element,
                        ]
                    )

            print(
                f"index: {(min_line, int(index_of_element))}\n"
                + f"focus func val: {round(simplex_matrix[-1, int(index_of_element)], ROUND_VAL)}\n"
                + f"focus val: {round(simplex_matrix[min_line, int(index_of_element)], ROUND_VAL)}",
            )
            print_matrix(simplex_matrix)

            simplex_matrix[min_line, :] = (
                simplex_matrix[min_line, :]
                / simplex_matrix[
                    min_line,
                    index_of_element,
                ]
            )

            for line in range(simplex_matrix.shape[0]):
                if line == min_line:
                    continue

                simplex_matrix[line, :] = (
                    simplex_matrix[line, :]
                    - simplex_matrix[min_line, :]
                    * simplex_matrix[line, index_of_element]
                )

    print("result: ")
    print_matrix(simplex_matrix)

    return simplex_matrix[-1, 0], simplex_matrix


def dual_simplex(simplex_matrix: np.ndarray):
    while True:
        index_of_element = simplex_matrix[:-1, 0].argmin()

        if simplex_matrix[:-1, 0][index_of_element] >= 0:
            break

        else:
            min_element = np.inf
            min_column = 0

            for column in range(1, simplex_matrix.shape[1]):
                if simplex_matrix[-1, column] == 0:
                    continue

                if (
                    simplex_matrix[index_of_element, column] < 0
                    and abs(
                        simplex_matrix[-1, column]
                        / simplex_matrix[index_of_element, column]
                    )
                    < min_element
                ):
                    min_column = column
                    min_element = abs(
                        simplex_matrix[-1, column]
                        / simplex_matrix[index_of_element, column]
                    )

            print(
                f"index: {(int(index_of_element), min_column)}\n"
                + f"focus func val: {round(simplex_matrix[:-1, 0][index_of_element], ROUND_VAL)}\n"
                + f"focus val: {round(simplex_matrix[int(index_of_element), min_column], ROUND_VAL)}",
            )
            print_matrix(simplex_matrix)

            simplex_matrix[index_of_element, :] /= simplex_matrix[
                index_of_element, min_column
            ]

            for line in range(simplex_matrix.shape[0]):
                if line == index_of_element:
                    continue

                simplex_matrix[line, :] -= (
                    simplex_matrix[index_of_element, :]
                    * simplex_matrix[line, min_column]
                )

    print("result: ")
    print_matrix(simplex_matrix)

    return simplex_matrix[-1, 0], simplex_matrix


A = np.array(
    [
        [15, 115, 106, 290, 232, 167],
        [79, 247, 7, 286, 65, 276],
        [219, 125, 174, 42, 114, 202],
        [287, 213, 225, 274, 169, 260],
        [202, 124, 211, 200, 174, 183],
        [158, 265, 1, 39, 113, 290],
        [175, 196, 170, 270, 187, 178],
        [245, 100, 226, 63, 245, 259],
    ]
)

b = np.array([296, 85, 22, 47, 247, 28, 125, 218])
c = np.array([173, 299, 240, 120, 249, 86])

print("simplex", end="\n\n")

x1 = simplex(make_matrix(A, b, -c))

print("dual simplex", end="\n\n")

x2 = dual_simplex(make_dual_matrix(A.T, b, -c))

print(f"simplex: {x1[0]}\ndual simplex: {x2[0]}\ndelta: {np.abs(x1[0] - x2[0])}\n")