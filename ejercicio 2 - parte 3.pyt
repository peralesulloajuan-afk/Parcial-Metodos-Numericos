import numpy as np
from scipy.interpolate import CubicSpline

# f: Frecuencias de excitación en kHz
f = np.array([
    10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 
    45.0, 47.5, 50.0, 52.5, 55.0, 57.5, 60.0, 62.5, 65.0, 67.5, 70.0, 72.5, 75.0, 77.5, 
    80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5
])

# V: Voltajes de salida medidos en Voltios
V = np.array([
    0.842, 0.911, 0.986, 1.062, 1.143, 1.227, 1.314, 1.401, 1.482, 1.551, 1.216, 1.048, 0.866, 0.689, 
    0.521, 0.364, 0.223, 0.103, 0.012, -0.041, -0.057, -0.034, 0.018, 0.096, 0.197, 0.318, 0.452, 0.579, 
    0.700, 0.809, 0.611, 0.688, 0.756, 0.811, 0.856, 0.894, 0.926, 0.954, 0.980, 1.004
])

# Construimos la función continua usando el Spline Cúbico Natural
cs_V = CubicSpline(f, V, bc_type='natural')

# 1. Búsqueda de intervalos de cambio de signo en los datos
sign_changes = []
for i in range(len(V)-1):
    if V[i] * V[i+1] < 0:
        sign_changes.append((f[i], f[i+1]))

print("1. Intervalos detectados:")
for idx, (a, b) in enumerate(sign_changes):
    print(f"   Cruce {idx+1}: Entre {a} kHz y {b} kHz")

# 2 y 3. Método de Bisección aplicado al Spline
def bisection(func, a, b, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        c = (a + b) / 2.0
        if abs(func(c)) < tol or (b - a) / 2.0 < tol:
            return c
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2.0

root1_bis = bisection(cs_V, sign_changes[0][0], sign_changes[0][1])
root2_bis = bisection(cs_V, sign_changes[1][0], sign_changes[1][1])

print("\n2 y 3. Método de Bisección (Tol = 1e-6):")
print(f"   Primera raíz: f = {root1_bis:.6f} kHz")
print(f"   Segunda raíz: f = {root2_bis:.6f} kHz")

# 4. Refinamiento extrayendo las raíces analíticas del Spline
all_roots = cs_V.roots()
valid_roots = [r for r in all_roots if f[0] <= r <= f[-1]]

print("\n4. Raíces Analíticas Exactas del Spline (Refinamiento):")
for idx, r in enumerate(valid_roots):
    print(f"   Raíz {idx+1}: f = {r:.6f} kHz")