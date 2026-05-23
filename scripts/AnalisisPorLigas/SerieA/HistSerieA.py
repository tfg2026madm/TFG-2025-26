import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dfSerieA = pd.read_csv(r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv")
SerieA = dfSerieA[dfSerieA["league"] == "La_liga"]

plt.hist(SerieA["xG"], bins=30, alpha=0.6, label="xG")
plt.hist(SerieA["scored"], bins=30, alpha=0.6, label="scored")
plt.hist(SerieA["xG_diff"], bins=30, alpha=0.6, label="xG_diff")

plt.legend()
plt.show()



