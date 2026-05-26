import matplotlib.pyplot as plt
import numpy as np

# Definición de los datos extraídos del experimento
# Frecuencias en Hz
f = np.array([100, 120, 145, 170, 200, 235, 270, 310, 355, 405, 460, 520, 
              585, 655, 730, 810, 895, 985, 1080, 1180, 1290, 1410, 1540, 
              1680, 1830, 1990, 2160, 2340, 2530, 2730])

# Magnitud de la impedancia |Z| en ohmios
Z = np.array([152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 
              133.6, 132.7, 131.9, 131.4, 131.1, 130.9, 131.0, 131.3, 131.9, 
              132.7, 133.8, 135.2, 136.9, 138.9, 141.1, 143.5, 146.1, 149.0, 
              152.2, 155.6, 159.2])

# Creación de la figura y configuración inicial
plt.figure(figsize=(10, 6))

# Graficamos los datos usando puntos discretos ('o') y una línea punteada ('--')
# para observar tanto la medición exacta como la tendencia continua.
plt.plot(f, Z, marker='o', linestyle='--', color='b', label='Datos medidos |Z|(f)')

# Identificación y marcado visual del mínimo local en la gráfica
min_index = np.argmin(Z)
plt.plot(f[min_index], Z[min_index], marker='o', color='r', markersize=10, 
         label=f'Mínimo local: ({f[min_index]:.4f} Hz, {Z[min_index]:.4f} ohm)')

# Añadimos etiquetas, título y cuadrícula para mejorar la calidad del gráfico
plt.title('Magnitud de Impedancia Bioeléctrica vs Frecuencia', fontsize=14)
plt.xlabel('Frecuencia f (Hz)', fontsize=12)
plt.ylabel('Magnitud de Impedancia |Z| (ohm)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()

# Mostramos el gráfico final
plt.show()
