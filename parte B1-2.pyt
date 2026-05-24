import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import random

# Datos de entrada
f = np.array([100, 120, 145, 170, 200, 235, 270, 310, 355, 405, 460, 520, 
              585, 655, 730, 810, 895, 985, 1080, 1180, 1290, 1410, 1540, 
              1680, 1830, 1990, 2160, 2340, 2530, 2730])
Z = np.array([152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 
              133.6, 132.7, 131.9, 131.4, 131.1, 130.9, 131.0, 131.3, 131.9, 
              132.7, 133.8, 135.2, 136.9, 138.9, 141.1, 143.5, 146.1, 149.0, 
              152.2, 155.6, 159.2])

# 1. Comparación y Fenómeno de Runge
f_eval = np.linspace(100, 2730, 500)
poly_29 = lagrange(f, Z) # Grado 29 (Global)
poly_5 = lagrange(f[12:18], Z[12:18])  # Grado 5 (Escalonado)
poly_10 = lagrange(f[10:21], Z[10:21]) # Grado 10 (Escalonado)
poly_15 = lagrange(f[7:23], Z[7:23])   # Grado 15 (Escalonado)

plt.figure(figsize=(8, 4))
plt.plot(f, Z, 'ko', label='Datos medidos')
plt.plot(f_eval, poly_5(f_eval), '--', label='Grado 5')
plt.plot(f_eval, poly_10(f_eval), '-.', label='Grado 10')
plt.ylim(120, 170) # Limitar eje Y para ver detalles
plt.plot(f_eval, poly_15(f_eval), ':', label='Grado 15')
plt.plot(f_eval, poly_29(f_eval), '-', color='red', alpha=0.5, label='Grado 29 (Runge)')
plt.title('Comparación de Polinomios y Fenómeno de Runge')
plt.xlabel('Frecuencia (Hz)'); plt.ylabel('|Z| (ohm)')
plt.legend(); plt.grid(True); plt.show()

# 2. Cálculo en f = 1000 Hz (Usando Grado 5 local)
idx = np.where((f >= 810) & (f <= 1290))[0] # Puntos cercanos a 1000 Hz
poly_sel = lagrange(f[idx], Z[idx])
print(f"|Z| en 1000 Hz: {poly_sel(1000):.4f} ohm")

# 3. Validación Leave-One-Out (LOO) con 5 puntos
random.seed(42)
indices_loo = random.sample(range(5, 25), 5) # Evitar bordes extremos
errores = []

for i in indices_loo:
    f_test, Z_true = f[i], Z[i]
    f_train, Z_train = np.delete(f, i), np.delete(Z, i)
    
    # Seleccionar los 6 puntos más cercanos para interpolación local (Grado 5)
    idx_cercanos = np.argsort(np.abs(f_train - f_test))[:6]
    poly_loo = lagrange(f_train[idx_cercanos], Z_train[idx_cercanos])
    
    error = abs((Z_true - poly_loo(f_test)) / Z_true) * 100
    errores.append(error)

print(f"Error Relativo Promedio (LOO): {np.mean(errores):.4f}%")