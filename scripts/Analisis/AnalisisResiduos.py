import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Cargar datos
# Asegúrate de que la ruta sea correcta en tu ordenador
file_path = r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv"
df = pd.read_csv(file_path)

# --- PASO 1: Agrupación de Datos ---
# Los datos originales son por partido. Para analizar la eficiencia de los equipos,
# debemos agrupar por 'team' y 'year' (temporada).
# Esto nos da el total de xG y Goles de cada equipo en cada temporada.
df_team_season = df.groupby(['league', 'year', 'team'])[['xG', 'scored']].sum().reset_index()

# --- PASO 2: Regresión Lineal ---
X = df_team_season[['xG']]
y = df_team_season[['scored']]

regr = linear_model.LinearRegression()
regr.fit(X, y)

# --- PASO 3: Cálculo de Residuos ---
# Predecimos cuántos goles "deberían" haber marcado según el modelo
df_team_season['scored_pred'] = regr.predict(X)

# El residuo es la diferencia: Realidad - Predicción
# Residuo Positivo (> 0): Marcaron MÁS de lo que el modelo predijo (Eficiencia/Suerte)
# Residuo Negativo (< 0): Marcaron MENOS de lo que el modelo predijo (Ineficiencia/Mala suerte)
df_team_season['residual'] = df_team_season['scored'] - df_team_season['scored_pred']

# Métricas del modelo global
r2 = regr.score(X, y)
m = regr.coef_[0][0]
b = regr.intercept_[0]

print(f"--- Resultados de la Regresión (Agrupado por Temporada) ---")
print(f"Pendiente (m): {m:.4f}")
print(f"Intercepto (b): {b:.4f}")
print(f"R^2: {r2:.4f}")

# --- PASO 4: Identificar Outliers ---
top_over = df_team_season.sort_values('residual', ascending=False).head(5)
top_under = df_team_season.sort_values('residual', ascending=True).head(5)

print("\n--- Top 5 'Overperformers' (Más eficientes) ---")
print(top_over[['team', 'year', 'league', 'scored', 'scored_pred', 'residual']])

print("\n--- Top 5 'Underperformers' (Menos eficientes) ---")
print(top_under[['team', 'year', 'league', 'scored', 'scored_pred', 'residual']])

# --- PASO 5: Visualización ---
plt.figure(figsize=(12, 8))

# Puntos de todos los equipos
plt.scatter(df_team_season['xG'], df_team_season['scored'], alpha=0.3, c='gray', label='Equipos (Temporadas)')

# Línea de regresión
plt.plot(df_team_season['xG'], df_team_season['scored_pred'], color='red', linewidth=2, label=f'Regresión (R^2={r2:.2f})')

# Resaltar Outliers
plt.scatter(top_over['xG'], top_over['scored'], color='green', s=100, label='Top Overperformers')
plt.scatter(top_under['xG'], top_under['scored'], color='blue', s=100, label='Top Underperformers')

# Etiquetas simples con alternancia de altura para evitar choques
for i, (idx, row) in enumerate(top_over.iterrows()):
    # Alternamos un poco la altura (3 o 6 unidades) para que no se pisen
    offset = 3 if i % 2 == 0 else 6
    plt.text(row['xG'], row['scored'] + offset, f"{row['team']} {row['year']}", 
             fontsize=9, ha='center', color='darkgreen', fontweight='bold')

for i, (idx, row) in enumerate(top_under.iterrows()):
    # Lo mismo hacia abajo
    offset = -3 if i % 2 == 0 else -6
    plt.text(row['xG'], row['scored'] + offset, f"{row['team']} {row['year']}", 
             fontsize=9, ha='center', va='top', color='darkblue', fontweight='bold')

plt.title('Análisis de Residuos: Eficiencia Goleadora por Temporada')
plt.xlabel('xG (Goles Esperados)')
plt.ylabel('Goles Reales')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
