import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')

# Cargar datos
file_path = r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv"
df = pd.read_csv(file_path)

# Separamos datos Locales (h) y visitantes (a)
df_home = df[df['h_a'] == 'h']
df_away = df[df['h_a'] == 'a']

# Calculamos el promedio de 'deep' (llegadas) para cada grupo
promedio_deep_home = df_home['deep'].mean()
promedio_deep_away = df_away['deep'].mean()

print(f"Promedio de llegadas locales: {promedio_deep_home:.2f}")
print(f"Promedio de llegadas visitantes: {promedio_deep_away:.2f}")

# Preparamos los datos para la gráfica
# En el Eje X: Los nombres de las barras
categorias = ['Local', 'Visitante']

# En el Eje Y: Las alturas de las barras (Promedios que calculamos)
valores = [promedio_deep_home, promedio_deep_away]

# Creamos la gráfica
plt.bar(categorias, valores, color=['blue', 'red'], alpha=0.7, width=0.5)
# 3. Añadir títulos y etiquetas
plt.title('Volumen de Juego: Llegadas a Zona de Peligro (Deep)', fontsize=14, fontweight='bold')
plt.ylabel('Promedio de Pases Profundos por Partido', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7) # Rejilla solo horizontal
plt.show()