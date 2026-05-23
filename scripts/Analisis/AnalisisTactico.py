import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import pearsonr

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')

# Cargar datos
file_path = r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv"
df = pd.read_csv(file_path)

# --- PASO 1: Agrupación de Datos ---
# Agrupamos por equipo y temporada para tener métricas consolidadas
# 'ppda_coef' es un promedio, 'deep', 'xG', 'scored' son sumas
df_tactical = df.groupby(['league', 'year', 'team']).agg({
    'xG': 'sum',
    'scored': 'sum',
    'deep': 'sum',
    'ppda_coef': 'mean' # Promedio de intensidad de presión
}).reset_index()

#league,year,team,xG,scored,deep,ppda_coef
#La_Liga,2017,Real Madrid,6.0,6,30,10.0

# Calculamos la Eficiencia (Goles - xG)
df_tactical['efficiency'] = df_tactical['scored'] - df_tactical['xG']

# --- PASO 2: Análisis de Correlaciones ---
corr_ppda = pearsonr(df_tactical['ppda_coef'], df_tactical['efficiency'])
corr_deep = pearsonr(df_tactical['deep'], df_tactical['scored'])

print(f"--- Correlaciones ---")
print(f"PPDA vs Eficiencia: r={corr_ppda[0]:.3f} (p-value={corr_ppda[1]:.4f})")
print(f"Deep vs Goles:      r={corr_deep[0]:.3f} (p-value={corr_deep[1]:.4f})")

# --- PASO 3: Visualización ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Gráfico 1: PPDA vs Eficiencia
# PPDA bajo = Presión Alta. PPDA alto = Presión Baja.
axes[0].scatter(df_tactical['ppda_coef'], df_tactical['efficiency'], alpha=0.3, color='purple')

# Línea de tendencia para PPDA
z = np.polyfit(df_tactical['ppda_coef'], df_tactical['efficiency'], 1)
p = np.poly1d(z)
axes[0].plot(df_tactical['ppda_coef'], p(df_tactical['ppda_coef']), "r--")

axes[0].set_title(f'Presión (PPDA) vs Eficiencia Goleadora\nCorrelación: {corr_ppda[0]:.2f}', fontsize=14)
axes[0].set_xlabel('PPDA (Pases permitidos por acción defensiva)\n<-- Más Presión | Menos Presión -->')
axes[0].set_ylabel('Eficiencia (Goles Reales - xG)')
axes[0].axhline(0, color='black', linestyle='--', alpha=0.5) # Línea de eficiencia neutra

# Gráfico 2: Deep vs Goles
axes[1].scatter(df_tactical['deep'], df_tactical['scored'], alpha=0.3, color='teal')

# Línea de tendencia para Deep
z2 = np.polyfit(df_tactical['deep'], df_tactical['scored'], 1)
p2 = np.poly1d(z2)
axes[1].plot(df_tactical['deep'], p2(df_tactical['deep']), "orange", linestyle='--')

axes[1].set_title(f'Llegadas a Zona de Peligro (Deep) vs Goles Totales\nCorrelación: {corr_deep[0]:.2f}', fontsize=14)
axes[1].set_xlabel('Deep (Pases completados cerca del área rival)')
axes[1].set_ylabel('Goles Totales')

plt.tight_layout()
plt.show()
