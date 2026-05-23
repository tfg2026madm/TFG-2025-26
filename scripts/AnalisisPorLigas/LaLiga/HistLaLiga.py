import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dfLaLiga = pd.read_csv(r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv")
La_liga = dfLaLiga[dfLaLiga["league"] == "La_liga"]

plt.hist(La_liga["xG"], bins=30, alpha=0.6, label="xG")
plt.hist(La_liga["scored"], bins=30, alpha=0.6, label="scored")
plt.hist(La_liga["xG_diff"], bins=30, alpha=0.6, label="xG_diff")

plt.legend()
plt.show()



