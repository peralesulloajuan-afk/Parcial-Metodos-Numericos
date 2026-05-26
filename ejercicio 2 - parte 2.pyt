import numpy as np
from scipy.interpolate import CubicSpline

# =========================================================
# 1. Definición de Datos Experimentales
# =========================================================
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

# h: Tamaño del paso constante (distancia entre frecuencias, 12.5 - 10.0 = 2.5)
h = 2.5

# =========================================================
# 2. Funciones de Derivación Numérica (Diferencias Finitas)
# =========================================================

# Diferencia centrada de orden 2: Error O(h^2)
# Fórmula: f'(x) = [f(x+h) - f(x-h)] / 2h
# Usa el punto siguiente (idx+1) y el anterior (idx-1).
def central_diff_O2(idx):
    return (-V[idx-1] + V[idx+1]) / (2*h)

# Diferencia centrada de orden 4: Error O(h^4)
# Fórmula: f'(x) = [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / 12h
# Es más precisa, pero exige conocer 2 puntos hacia adelante y 2 hacia atrás.
def central_diff_O4(idx):
    return (V[idx-2] - 8*V[idx-1] + 8*V[idx+1] - V[idx+2]) / (12*h)

# Diferencia progresiva de orden 2: Error O(h^2)
# Fórmula: f'(x) = [-3f(x) + 4f(x+h) - f(x+2h)] / 2h
# Se usa en el extremo izquierdo (f=10 kHz) donde no tenemos datos "hacia atrás".
def forward_diff_O2(idx):
    return (-3*V[idx] + 4*V[idx+1] - V[idx+2]) / (2*h)

# =========================================================
# 3. Búsqueda de Índices
# =========================================================
# Usamos np.where para localizar en qué posición (índice) del arreglo 'f' 
# se encuentran las frecuencias que nos pide el problema.
indices = {40.0: np.where(f==40.0)[0][0], 70.0: np.where(f==70.0)[0][0], 100.0: np.where(f==100.0)[0][0]}

# =========================================================
# 4. Cálculo y Resultados: Diferencias Centradas
# =========================================================
print("1. Diferencia Central")
for freq, idx in indices.items():
    # Iteramos sobre las frecuencias y aplicamos ambas funciones de orden 2 y orden 4.
    # El modificador :.6f asegura que se impriman exactamente 6 decimales.
    print(f"f = {freq}: O(h^2) = {central_diff_O2(idx):.6f}, O(h^4) = {central_diff_O4(idx):.6f}")

# =========================================================
# 5. Cálculo y Resultados: Diferencia Progresiva
# =========================================================
# Calculamos para el borde inferior de la tabla (10.0 kHz)
idx_10 = np.where(f==10.0)[0][0]
print(f"\n2. Diferencia Progresiva en f=10.0: O(h^2) = {forward_diff_O2(idx_10):.6f}")

# =========================================================
# 6. Derivación Analítica con Spline Cúbico Natural
# =========================================================
# Ajustamos un Spline Cúbico Natural (bc_type='natural') a todos los datos.
cs_V = CubicSpline(f, V, bc_type='natural')

# Extraemos la ecuación analítica exacta de su primera derivada.
cs_V_diff = cs_V.derivative()

print("\n4. Derivada del Spline")
# Como el Spline devuelve una función matemática continua, ya no necesitamos índices,
# podemos evaluarlo pasándole directamente la frecuencia objetivo.
for freq in [10.0, 40.0, 70.0, 100.0]:
    print(f"f = {freq}: Spline = {cs_V_diff(freq):.6f}")