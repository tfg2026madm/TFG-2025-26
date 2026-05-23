import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dfBundes = pd.read_csv(r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv")
Bundesliga = dfBundes[dfBundes["league"] == "Bundesliga"]

plt.hist(Bundesliga["xG"], bins=30, alpha=0.6, label="xG")
plt.hist(Bundesliga["scored"], bins=30, alpha=0.6, label="scored")
plt.hist(Bundesliga["xG_diff"], bins=30, alpha=0.6, label="xG_diff")

plt.legend()
plt.show()



