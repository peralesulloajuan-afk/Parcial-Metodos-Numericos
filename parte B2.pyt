import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, lagrange

# 1. Datos originales del experimento
f = np.array([100, 120, 145, 170, 200, 235, 270, 310, 355, 405, 460, 520, 
              585, 655, 730, 810, 895, 985, 1080, 1180, 1290, 1410, 1540, 
              1680, 1830, 1990, 2160, 2340, 2530, 2730])
Z = np.array([152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 
              133.6, 132.7, 131.9, 131.4, 131.1, 130.9, 131.0, 131.3, 131.9, 
              132.7, 133.8, 135.2, 136.9, 138.9, 141.1, 143.5, 146.1, 149.0, 
              152.2, 155.6, 159.2])

# 2. Construcción del Spline Cúbico Natural
# bc_type='natural' asegura que la segunda derivada en los extremos sea cero.
cs_natural = CubicSpline(f, Z, bc_type='natural')

# 3. Recrear el Polinomio Local (Grado 5, seleccionado en B1) para comparación
idx_local = np.where((f >= 810) & (f <= 1290))[0]
poly_local = lagrange(f[idx_local], Z[idx_local])

# 4. Evaluación en Malla Fina
f_fina = np.linspace(100, 2730, 500)
Z_spline = cs_natural(f_fina)

# Evaluamos el polinomio local solo en una zona cercana a su dominio 
# para que no deforme el gráfico al divergir.
f_fina_local = np.linspace(500, 1600, 200)
Z_poly_local = poly_local(f_fina_local)

# 5. Gráfica comparativa
plt.figure(figsize=(9, 5))
plt.plot(f, Z, 'ko', label='Datos medidos')
plt.plot(f_fina, Z_spline, 'b-', label='Spline Cúbico Natural')
plt.plot(f_fina_local, Z_poly_local, 'r--', label='Polinomio Local (Grado 5)')

plt.title('Comparación: Spline Cúbico vs Polinomio Local')
plt.xlabel('Frecuencia f (Hz)')
plt.ylabel('Magnitud de Impedancia |Z| (ohm)')
plt.xlim(400, 1700) # Zoom a la zona de interés (alrededor de 1000 Hz)
plt.ylim(130, 145)
plt.legend()
plt.grid(True)
plt.show() # Recuerda cerrar la ventana para que se impriman los resultados

# 6. Cálculo en f = 1000 Hz y comparación
Z_1000_spline = cs_natural(1000)
Z_1000_poly = poly_local(1000)

print("\n--- Resultados en f = 1000 Hz ---")
print(f"Valor interpolado (Spline Cúbico Natural): {Z_1000_spline:.4f} ohm")
print(f"Valor interpolado (Polinomio Grado 5)    : {Z_1000_poly:.4f} ohm")
print(f"Diferencia absoluta entre métodos        : {abs(Z_1000_spline - Z_1000_poly):.4f} ohm")