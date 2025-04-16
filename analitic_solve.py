import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def circulation_cylinder(a, alpha, vinf, gamma, size):
    # Центр цилиндра
    cx, cy = 0.5, 0.5  
    c = a  # Радиус цилиндра
    alpha_rad = np.radians(alpha)  # Угол атаки в радианах
    
    # Функции потенциала и скорости
    def velocity_field(x, y):
        z = (x - cx) + 1j * (y - cy)
        if np.abs(z) < c:  # Если точка внутри цилиндра, возвращаем NaN
            return np.nan, np.nan
        w_prime = vinf * (np.exp(-1j * alpha_rad) - c**2 / z**2 * np.exp(1j * alpha_rad))
        return np.real(w_prime), -np.imag(w_prime)
    
    # Поиск стагнационных точек
    def stagnation_eq(x):
        """ Уравнение для нахождения стагнационной точки (по горизонтальной оси) """
        x = x[0]  # Берём первый элемент массива
        z = complex(x - cx, cy - cy)  # Только x-координата меняется, y=cy
        w_prime = vinf * (np.exp(-1j * alpha_rad) - c**2 / z**2 * np.exp(1j * alpha_rad))
        return np.real(w_prime)  # Должно быть равно 0

    # Ищем два корня (левый и правый стагнационные точки)
    stagnation_x1 = fsolve(stagnation_eq, cx + c)[0]
    stagnation_x2 = fsolve(stagnation_eq, cx - c)[0]
    stagnation_points = np.array([[stagnation_x1, cy], [stagnation_x2, cy]])
    
    # Вычисление циркуляции в стагнационных точках
    def circulation_at_stagnation(x, y):
        z = complex(x - cx, y - cy)
        w_prime = vinf * (np.exp(-1j * alpha_rad) - c**2 / z**2 * np.exp(1j * alpha_rad)) + 1j * gamma / (2 * np.pi * z)
        return np.imag(w_prime)  # Функция тока в стагнационной точке

    # Суммарная циркуляция по двум стагнационным точкам
    total_circulation = 0
    for point in stagnation_points:
        circulation_value = circulation_at_stagnation(point[0], point[1])
        total_circulation += circulation_value
    
    # Вывод стагнационных точек
    for i, point in enumerate(stagnation_points):
        print(f"Критическая точка {i+1}: X = {point[0]}, Y = {point[1]}")
    
    # Вывод суммарной циркуляции
    print(f"Циркуляция: {total_circulation}")
    
    # Создание сетки
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Вычисление поля скоростей
    U, V = np.vectorize(velocity_field)(X, Y)

    # Вычисление скорости
    speed = np.sqrt(np.abs(U)**2 + np.abs(V)**2)
    
    # График
    fig, ax = plt.subplots(figsize=(10, 10))

    # Цветовая подложка
    plt.tricontourf(X, Y, speed, shading='auto', cmap='viridis') 

    # Добавляем цветовую шкалу
    plt.colorbar(label='Модуль вектора скорости')
    
    # Линии тока
    stream = ax.streamplot(X, Y, U, V, color='black', linewidth=1.2, density=1.8, minlength=0.85, arrowsize=0)
    
    # Отрисовка цилиндра
    theta = np.linspace(0, 2 * np.pi, 300)
    cylinder_x = cx + c * np.cos(theta)
    cylinder_y = cy + c * np.sin(theta)
    ax.fill(cylinder_x, cylinder_y, 'w', zorder=3)
    cylinder_line, = ax.plot(cylinder_x, cylinder_y, 'r', linewidth=2.2, zorder=4)
    
    # Отметка стагнационных точек
    stagnation_plot = ax.scatter(stagnation_points[:, 0], stagnation_points[:, 1], color='lime', s=100, zorder=5)

    # Настройка осей
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Обтекание цилиндра с циркулляцией")
    
    # Добавляем легенду
    ax.legend(handles=[stagnation_plot, cylinder_line, stream.lines], labels=["Критические точки", "Контур цилиндра", "Линии тока"])

    # Сохраняем изображение в файл PNG
    plt.savefig("circulation_cylinder.png", format="png")
    plt.close(fig)  # Закрываем фигуру

# Вызов функции
circulation_cylinder(0.125, 0, 1.0, 0.0, 1000)
