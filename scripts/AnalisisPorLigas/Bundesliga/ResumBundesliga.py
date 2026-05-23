import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# -> from sklearn import linear_model
# -> from scipy import stats

dfBundes = pd.read_csv(r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv")
df_years = dfBundes.groupby("year")
Filtr = dfBundes[dfBundes["league"] == "Bundesliga"]

media_goles = dfBundes.groupby("year")[['xG', 'scored', 'xG_diff']].mean()
partidos_por_año = dfBundes.groupby("year").size()

tabla = Filtr.groupby("year").agg(
    partidos = ('team', 'count'),
    media_goles_esperados = ('xG', 'mean'),
    media_goles_marcados = ('scored', 'mean'),
    media_diff_goles = ('xG_diff', 'mean')
)

print(tabla)

tabla.to_csv("Bundesliga_resumen.csv", index=False)



