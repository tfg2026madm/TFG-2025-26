import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dfUstat = pd.read_csv(r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv")

plt.hist(dfUstat["xG"], bins=30, alpha=0.6, label="xG")
plt.hist(dfUstat["scored"], bins=30, alpha=0.6, label="scored")
plt.hist(dfUstat["xG_diff"], bins=30, alpha=0.6, label="xG_diff")

plt.legend()
plt.show()



