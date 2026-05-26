import numpy as np
from scipy.interpolate import lagrange, CubicSpline

# 40 datos de frecuencia (kHz), voltaje (V) y módulo de impedancia (ohm)
f = np.array([10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0, 57.5, 60.0, 62.5, 65.0, 67.5, 70.0, 72.5, 75.0, 77.5, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5])
V = np.array([0.842, 0.911, 0.986, 1.062, 1.143, 1.227, 1.314, 1.401, 1.482, 1.551, 1.216, 1.048, 0.866, 0.689, 0.521, 0.364, 0.223, 0.103, 0.012, -0.041, -0.057, -0.034, 0.018, 0.096, 0.197, 0.318, 0.452, 0.579, 0.700, 0.809, 0.611, 0.688, 0.756, 0.811, 0.856, 0.894, 0.926, 0.954, 0.980, 1.004])
Z = np.array([182.4, 178.9, 175.1, 171.0, 166.8, 162.7, 158.9, 155.4, 152.0, 149.0, 146.1, 145.2, 145.8, 147.3, 149.9, 153.5, 158.0, 163.2, 168.9, 174.8, 180.5, 186.2, 191.5, 196.2, 200.1, 203.1, 205.2, 206.3, 206.1, 204.7, 198.0, 194.4, 190.9, 187.8, 185.1, 183.0, 181.6, 180.8, 180.6, 180.9])

f_eval = [41.0, 73.0]

print("--- 1. INTERPOLACIÓN DE LAGRANGE (GRADO 2) ---")
for fe in f_eval:
    # Buscamos los 3 puntos más cercanos a la frecuencia deseada
    idx_cercanos = np.argsort(np.abs(f - fe))[:3]
    
    # Construimos el polinomio local de grado 2
    poly_V = lagrange(f[idx_cercanos], V[idx_cercanos])
    poly_Z = lagrange(f[idx_cercanos], Z[idx_cercanos])
    
    print(f"Estimación en f = {fe} kHz:")
    print(f"  V({fe}) = {poly_V(fe):.4f} V")
    print(f"  |Z|({fe}) = {poly_Z(fe):.4f} ohm\n")

print("--- 2. SPLINE CÚBICO NATURAL ---")
# Construimos los splines usando toda la tabla de datos
cs_V = CubicSpline(f, V, bc_type='natural')
cs_Z = CubicSpline(f, Z, bc_type='natural')

for fe in f_eval:
    print(f"Estimación en f = {fe} kHz:")
    print(f"  V({fe}) = {cs_V(fe):.4f} V")
    print(f"  |Z|({fe}) = {cs_Z(fe):.4f} ohm\n")