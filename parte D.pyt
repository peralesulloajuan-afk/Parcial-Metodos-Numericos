# Definimos la función objetivo g(f) = |Z|(f) - 150 y su derivada analítica
import numpy as np
from scipy.interpolate import CubicSpline

f = np.array([100, 120, 145, 170, 200, 235, 270, 310, 355, 405, 460, 520, 
              585, 655, 730, 810, 895, 985, 1080, 1180, 1290, 1410, 1540, 
              1680, 1830, 1990, 2160, 2340, 2530, 2730])
Z = np.array([152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 
              133.6, 132.7, 131.9, 131.4, 131.1, 130.9, 131.0, 131.3, 131.9, 
              132.7, 133.8, 135.2, 136.9, 138.9, 141.1, 143.5, 146.1, 149.0, 
              152.2, 155.6, 159.2])

# Construcción del Spline y su Segunda Derivada Analítica
cs_natural = CubicSpline(f, Z, bc_type='natural')

def g(x): return cs_natural(x) - 150
def dg(x): return cs_natural.derivative(nu=1)(x)

# 1. Método de Bisección
def biseccion(a, b, tol=1e-6):
    iters = 0
    while (b - a) / 2.0 > tol:
        c = (a + b) / 2.0
        if g(c) == 0: return c, iters
        if g(a) * g(c) < 0: b = c
        else: a = c
        iters += 1
    return (a + b) / 2.0, iters

# 2. Método de Newton-Raphson
def newton(x0, tol=1e-6):
    iters, x = 0, x0
    while True:
        x_new = x - g(x) / dg(x)
        iters += 1
        if abs(x_new - x) < tol: return x_new, iters
        x = x_new

# Ejecución en las zonas de cruce (observadas visualmente o en tabla)
# Primer límite (caída inicial): entre 100 y 120 Hz
f1_b, iter1_b = biseccion(100, 120)
f1_n, iter1_n = newton(110)

# Segundo límite (subida posterior, cerca de 2000 Hz): entre 2100 y 2400 Hz
f2_b, iter2_b = biseccion(2100, 2400)
f2_n, iter2_n = newton(2200)

print(f"Límite Inferior: {f1_n:.4f} Hz | Bisección: {iter1_b} iters, Newton: {iter1_n} iters")
print(f"Límite Superior: {f2_n:.4f} Hz | Bisección: {iter2_b} iters, Newton: {iter2_n} iters")

# 3. Cálculo de Sensibilidad en el límite superior
sensibilidad = 1.0 / dg(f2_n)
print(f"Sensibilidad df/d|Z| en Límite Superior: {sensibilidad:.4f} Hz/ohm")