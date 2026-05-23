import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dfPremier = pd.read_csv(r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv")
Premier = dfPremier[dfPremier["league"] == "EPL"]

plt.hist(Premier["xG"], bins=30, alpha=0.4, label="xG")
plt.hist(Premier["scored"], bins=30, alpha=0.4, label="scored")
plt.hist(Premier["xG_diff"], bins=30, alpha=0.4, label="xG_diff")

plt.legend()
plt.show()



