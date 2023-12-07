# import numpy as np
#
#
# def f(x):
#     return (x[0] - 3 * x[1]) ** 2 + (x[1] + 1) ** 2
#
#
# # def phi(alpha, x, direction):
# #     x1 = x[0] + alpha * direction[0]
# #     x2 = x[1] + alpha * direction[1]
# #     return f([x1, x2])
# #
# #
# # def binary_search_method(func, x, t, e, direction):
# #     h = 1
# #     x0 = x
# #     fx0 = func(x0, t, direction)
# #     while np.abs(h) > e:
# #         x1 = x0 + h
# #         fx1 = func(x1, t, direction)
# #         if fx0 < fx1:
# #             h /= -4
# #         x0 = x1
# #         fx0 = fx1
# #     return x0
# #
# #
# # def cyclic_coordinate_descent(func, x0, e):
# #     x = x0
# #     n = len(x)
# #     while True:
# #         x_prev = x.copy()
# #         for i in range(n):
# #             direction = np.zeros(n)
# #             direction[i] = 1.0
# #             alpha_opt = binary_search_method(phi, 1, x, e, direction)
# #             #print(alpha_opt)
# #             x += alpha_opt * direction
# #             # print(x)
# #             r = func(x)
# #             r1 = func(x_prev)
# #             if 0<r<1:
# #                 g=0
# #         if np.linalg.norm(x - x_prev) < e or np.abs(func(x) - func(x_prev)) < e:
# #             break
# #     return x
#
# def cyclic_coordinate_descent(func, x0, e, alpha):
#     x = x0
#     x_prev = x
#     k = 0
#     while np.linalg.norm(x - x_prev) < e or np.abs(func(x) - func(x_prev)) < e:
#         y = np.round(x - alpha * gradient_f(x), 15)
#         if func(y) > func(x):
#             alpha /= 2
#         else:
#             x = y
#             g = gradient_f(x)
#         k += 1
#     return x
#
# x0 = np.array([0.0, 8.0])
# e = 0.15
# solution = cyclic_coordinate_descent(f, x0, e)
# print('Точка минимума функции:', solution)
# print('Значение функции в этой точке:', f(solution))

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.linalg import LinAlgError


def func(x):
    return (x[0] - 3 * x[1]) ** 2 + (x[1] + 1) ** 2


x0 = np.array([0.0, 0.0])


def powell(f, x, tol=0.25, max_iter=100):
    # создание единичных векторов
    directions = np.eye(len(x))

    for _ in range(max_iter):
        x_prev = x.copy()

        for d in directions:
            # Поиск ti методом поразрядного поиска
            ti = minimize_scalar(lambda ti: f(x + ti * d)).x
            x = x + ti * d

        # Смена направлений поиска
        directions = np.roll(directions, shift=-1, axis=0)
        directions[-1] = x - x_prev

        try:
            # Проверка ранга
            rank = np.linalg.matrix_rank(directions)
        except LinAlgError:
            rank = len(directions) + 1

        if rank == len(directions):
            x = x + minimize_scalar(lambda ti: f(x + ti * directions[-1])).x * directions[-1]
        else:
            # Если ранг не полный, пересчитываем направления
            directions[-1] = x - x_prev

        if np.linalg.norm(x - x_prev) < tol:
            break

    return x


result = powell(func, x0)
print(f"Optimal point: ({result[0]:.30f}, {result[1]:.30f})")
print(f'Значение функции в точке минимума:{func(result):.30f}')
