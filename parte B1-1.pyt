import numpy as np
from scipy.interpolate import lagrange

# Subconjunto de 6 puntos centrados alrededor de 1000 Hz
f_local = np.array([810, 895, 985, 1080, 1180, 1290])
Z_local = np.array([131.0, 131.3, 131.9, 132.7, 133.8, 135.2])
f_objetivo = 1000

print("--- Interpolación Local (Grado 5) en f = 1000 Hz ---")

# ==========================================
# 1. MÉTODO MATRICIAL
# ==========================================
# Construimos la matriz de Vandermonde V de tamaño 6x6
# La ecuación es V * c = Z
V = np.vander(f_local, increasing=True)

# Resolvemos el sistema lineal para hallar los coeficientes del polinomio 'c'
coeficientes = np.linalg.solve(V, Z_local)

# Evaluamos el polinomio resultante en 1000 Hz
# Z = c0 + c1*(f) + c2*(f^2) + c3*(f^3) + c4*(f^4) + c5*(f^5)
potencias = np.array([f_objetivo**i for i in range(6)])
Z_matricial = np.dot(coeficientes, potencias)

print(f"Valor estimado (Método Matricial) : {Z_matricial:.4f} ohm")

# ==========================================
# 2. MÉTODO DE LAGRANGE
# ==========================================
# Construimos el polinomio usando la suma ponderada de funciones base de Lagrange
poly_lagrange = lagrange(f_local, Z_local)

# Evaluamos en 1000 Hz
Z_lagrange = poly_lagrange(f_objetivo)

print(f"Valor estimado (Método Lagrange)  : {Z_lagrange:.4f} ohm")