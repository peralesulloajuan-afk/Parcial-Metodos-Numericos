import numpy as np
from scipy.interpolate import CubicSpline

# Datos del experimento
f = np.array([100, 120, 145, 170, 200, 235, 270, 310, 355, 405, 460, 520, 
              585, 655, 730, 810, 895, 985, 1080, 1180, 1290, 1410, 1540, 
              1680, 1830, 1990, 2160, 2340, 2530, 2730])
Z = np.array([152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 
              133.6, 132.7, 131.9, 131.4, 131.1, 130.9, 131.0, 131.3, 131.9, 
              132.7, 133.8, 135.2, 136.9, 138.9, 141.1, 143.5, 146.1, 149.0, 
              152.2, 155.6, 159.2])

# Construcción del Spline y su Segunda Derivada Analítica
cs_natural = CubicSpline(f, Z, bc_type='natural')
cs_d2 = cs_natural.derivative(nu=2) # nu=2 indica segunda derivada

# Frecuencia del mínimo encontrada en la Parte C1
f_min = 742.1585 

# Evaluación de la segunda derivada en el mínimo
d2Z_df2 = cs_d2(f_min)

print(f"Segunda derivada en {f_min} Hz: {d2Z_df2:.6f} ohm/Hz^2")
