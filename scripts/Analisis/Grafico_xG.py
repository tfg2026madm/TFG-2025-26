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

sns.barplot(data=df_ligas, x="league", y="xG", palette="Greens", errorbar=None)

#plt.title("Media de Goles Esperados (xG) por Partido según la Liga", fontsize=15, fontweight='bold', pad=15)
plt.xlabel("Competición", fontsize=12)
plt.ylabel("Goles Esperados - xG (Media)", fontsize=12)

plt.tight_layout()
plt.show()