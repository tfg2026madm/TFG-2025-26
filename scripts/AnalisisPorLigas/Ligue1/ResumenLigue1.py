import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# -> from sklearn import linear_model
# -> from scipy import stats

dfLigue1 = pd.read_csv(r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv")
df_years = dfLigue1.groupby("year")
Filtr = dfLigue1[dfLigue1["league"] == "Ligue_1"]

media_goles = dfLigue1.groupby("year")[['xG', 'scored', 'xG_diff']].mean()
partidos_por_año = dfLigue1.groupby("year").size()

tabla = Filtr.groupby("year").agg(
    partidos = ('team', 'count'),
    media_goles_esperados = ('xG', 'mean'),
    media_goles_marcados = ('scored', 'mean'),
    media_diff_goles = ('xG_diff', 'mean')
)

print(tabla)

tabla.to_csv("Ligue1_resumen.csv", index=False)



