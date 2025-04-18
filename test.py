import numpy as np
import matplotlib.pyplot as plt

# Параметры
U = 1.0    # скорость потока
R = 1.0    # радиус цилиндра

# Создание сетки
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Комплексный потенциал (исключаем точки внутри цилиндра)
mask = np.abs(Z) >= R
W = np.full_like(Z, np.nan, dtype=np.complex128)
W[mask] = U * (Z[mask] + R**2 / Z[mask])

# Разделение на φ и ψ
phi = np.real(W)
psi = np.imag(W)

# Визуализация
plt.figure(figsize=(10, 10))
contours_psi = plt.contour(X, Y, psi, levels=50, colors='blue', linewidths=0.8)
contours_phi = plt.contour(X, Y, phi, levels=50, colors='red', linewidths=0.8)
plt.gca().set_aspect('equal')
circle = plt.Circle((0, 0), R, color='black', fill=False, linewidth=2)
plt.gca().add_patch(circle)

plt.title("Потенциальное обтекание цилиндра")
plt.xlabel("x")
plt.ylabel("y")
plt.legend([contours_psi.collections[0], contours_phi.collections[0], circle],
           ["Линии тока (ψ)", "Потенциал (φ)", "Цилиндр"])
plt.grid(True)
plt.tight_layout()
plt.show()