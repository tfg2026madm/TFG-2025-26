import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Configuración de fuentes y estilo para LaTeX 
plt.style.use('seaborn-v0_8-whitegrid')

# Aumentamos los tamaños base para que al escalar en LaTeX se lean bien
plt.rcParams.update({
    'font.size': 12,         
    'axes.titlesize': 14,     
    'axes.labelsize': 12,     
    'xtick.labelsize': 11,    
    'ytick.labelsize': 11,    
    'legend.fontsize': 12,
    'figure.titlesize': 16
})

# Cargar datos 
file_path = r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv"
df = pd.read_csv(file_path)

# Agrupación de Datos
df_tactical = df.groupby(['league', 'year', 'team']).agg({
    'xG': 'sum',
    'scored': 'sum',
    'deep': 'sum',
    'ppda_coef': 'mean' 
}).reset_index()

df_tactical['efficiency'] = df_tactical['scored'] - df_tactical['xG']

# Análisis de Correlaciones 
corr_ppda = pearsonr(df_tactical['ppda_coef'], df_tactical['efficiency'])
corr_deep = pearsonr(df_tactical['deep'], df_tactical['scored'])

print(f"--- Correlaciones ---")
print(f"PPDA vs Eficiencia: r={corr_ppda[0]:.3f} (p-value={corr_ppda[1]:.4f})")
print(f"Deep vs Goles:      r={corr_deep[0]:.3f} (p-value={corr_deep[1]:.4f})")

# Visualización 
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Gráfico 1: PPDA vs Eficiencia
axes[0].scatter(df_tactical['ppda_coef'], df_tactical['efficiency'], alpha=0.3, color='purple', s=40) # s=40 aumenta el tamaño del punto

z = np.polyfit(df_tactical['ppda_coef'], df_tactical['efficiency'], 1)
p = np.poly1d(z)
axes[0].plot(df_tactical['ppda_coef'], p(df_tactical['ppda_coef']), "r--", linewidth=2) 

axes[0].set_title(f'Presión (PPDA) vs Eficiencia Goleadora\n(r={corr_ppda[0]:.2f})')
axes[0].set_xlabel('PPDA\n<-- Más Presión | Menos Presión -->')
axes[0].set_ylabel('Eficiencia (Goles - xG)')
axes[0].axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)

# Gráfico 2: Deep vs Goles
axes[1].scatter(df_tactical['deep'], df_tactical['scored'], alpha=0.3, color='teal', s=40)

z2 = np.polyfit(df_tactical['deep'], df_tactical['scored'], 1)
p2 = np.poly1d(z2)
axes[1].plot(df_tactical['deep'], p2(df_tactical['deep']), "orange", linestyle='--', linewidth=2)

axes[1].set_title(f'Llegadas a Zona de Peligro (Deep) vs Goles\n(r={corr_deep[0]:.2f})')
axes[1].set_xlabel('Deep (Pases completados cerca del área rival)')
axes[1].set_ylabel('Goles Totales')

plt.tight_layout()


plt.show()