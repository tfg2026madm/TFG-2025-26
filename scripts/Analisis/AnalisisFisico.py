import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')

# Cargar datos
file_path = r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv"
df = pd.read_csv(file_path)

# Calculamos el PPDA medio por equipo y por liga para cada temporada
df_teams = df.groupby(['league', 'team', 'year'])['ppda_coef'].mean().reset_index()

# Ordenamos las ligas por su mediana de presión (ppda_coef)
orden_ligas = df.groupby(['league'])['ppda_coef'].median().sort_values().index

# Creamos el gráfico de cajas
plt.figure(figsize=(12, 7))

# Paletas de colores (tonos más oscuros para ligas más intensas)
ax = sns.boxplot(x='league', y='ppda_coef', data=df_teams, order=orden_ligas, palette='YlOrRd_r')

# Personalizamos el gráfico
#plt.title('Intensidad Física y Presión por Ligas en Europa (Métrica PPDA)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Liga Europea', fontsize=14)
plt.ylabel('PPDA (Pases permitidos por acción defensiva)', fontsize=14)

#plt.annotate('⬇ Mayor Intensidad Física / Presión Alta ⬇', 
#             xy=(0.5, 0.05), xycoords='axes fraction', 
#             ha='center', fontsize=12, color='darkred', weight='bold',
#             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=2))

plt.show()

for liga in df_teams['league'].unique():
    datos_liga = df_teams[df_teams['league'] == liga]

    Q1 = datos_liga['ppda_coef'].quantile(0.25)
    Q3 = datos_liga['ppda_coef'].quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = datos_liga[(datos_liga['ppda_coef'] < limite_inferior) | (datos_liga['ppda_coef'] > limite_superior)]

    if not outliers.empty:
        for index, row in outliers.iterrows():
            if row['ppda_coef'] < limite_inferior:
                tipo = "Presión EXTREMADAMENTE ALTA"
            else:
                tipo = "Presión EXTREMADAMENTE BAJA"
            
            print(f"  {row['team']} ({row['year']}): PPDA = {row['ppda_coef']:.2f} -> {tipo}")