import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
ruta = r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv"
df = pd.read_csv(ruta)

# Filtrar las ligas
ligas = ["Bundesliga", "EPL", "La_liga", "Ligue_1", "RFPL", "Serie_A"]
df_ligas = df[df["league"].isin(ligas)]

# Configurar y crear el gráfico
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

sns.barplot(data=df_ligas, x="league", y="xG_diff", palette="flare", errorbar=None)

# Añadir línea de referencia en el 0
plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='xG_diff = 0')

#plt.title("Media de la Diferencia de Goles (xG_diff) según la Liga", fontsize=15, fontweight='bold', pad=15)
plt.xlabel("Competición", fontsize=12)
plt.ylabel("Diferencia (xG_diff)", fontsize=12)
plt.legend()

plt.tight_layout()
plt.show()