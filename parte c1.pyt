import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Datos del experimento
f = np.array([100, 120, 145, 170, 200, 235, 270, 310, 355, 405, 460, 520, 
              585, 655, 730, 810, 895, 985, 1080, 1180, 1290, 1410, 1540, 
              1680, 1830, 1990, 2160, 2340, 2530, 2730])
Z = np.array([152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 
              133.6, 132.7, 131.9, 131.4, 131.1, 130.9, 131.0, 131.3, 131.9, 
              132.7, 133.8, 135.2, 136.9, 138.9, 141.1, 143.5, 146.1, 149.0, 
              152.2, 155.6, 159.2])

# 1. Construcción del Spline y su Derivada Analítica
cs_natural = CubicSpline(f, Z, bc_type='natural')
cs_derivada = cs_natural.derivative(nu=1) # nu=1 indica primera derivada

# 2. Evaluar la derivada en todos los puntos de datos originales
dZdf_puntos = cs_derivada(f)

# 3. Encontrar la frecuencia exacta del mínimo (donde derivada = 0)
raices = cs_derivada.roots()
# Filtramos la raíz que cae dentro de nuestro rango experimental
raiz_valida = [r for r in raices if np.isreal(r) and 100 <= r <= 2730][0]
raiz_valida = np.real(raiz_valida)

# 4. Generar la gráfica
f_fina = np.linspace(100, 2730, 500)
dZdf_fina = cs_derivada(f_fina)

plt.figure(figsize=(9, 5))
# Graficamos la curva continua de la derivada
plt.plot(f_fina, dZdf_fina, 'b-', label='Derivada Analítica del Spline')
# Superponemos los puntos exactos de los datos
plt.plot(f, dZdf_puntos, 'ko', label='Derivada en puntos de datos')
# Línea cero para evidenciar el cambio de signo
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
# Marcamos el cruce exacto
plt.plot(raiz_valida, 0, 'ro', markersize=8, 
         label=f'Cruce por cero: {raiz_valida:.4f} Hz')

plt.title('Derivada de la Impedancia respecto a la Frecuencia')
plt.xlabel('Frecuencia f (Hz)')
plt.ylabel('Derivada d|Z|/df (ohm/Hz)')
plt.legend()
plt.grid(True)
plt.show()

# Imprimir resultados
print(f"La derivada cambia de negativa a positiva exactamente en: {raiz_valida:.4f} Hz")