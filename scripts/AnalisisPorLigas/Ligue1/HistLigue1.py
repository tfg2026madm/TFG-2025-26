import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dfLigue1 = pd.read_csv(r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv")
Ligue1 = dfLigue1[dfLigue1["league"] == "Ligue_1"]

plt.hist(Ligue1["xG"], bins=30, alpha=0.6, label="xG")
plt.hist(Ligue1["scored"], bins=30, alpha=0.6, label="scored")
plt.hist(Ligue1["xG_diff"], bins=30, alpha=0.6, label="xG_diff")

plt.legend()
plt.show()



