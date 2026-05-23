import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

dfPremier = pd.read_csv(r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv")
Premier = dfPremier[dfPremier["league"] == "EPL"] 

xG = Premier[['xG']]
scored = Premier[['scored']]

regr = linear_model.LinearRegression()
regr.fit(xG, scored)

x_line = np.linspace(xG.min(), xG.max(), 100).reshape(-1, 1)
y_line = regr.predict(x_line)

plt.scatter(xG, scored, alpha=0.2)
plt.plot(x_line, y_line, color="red", label="Regresión lineal")

plt.xlabel("xG (goles esperados)")
plt.ylabel("scored (goles reales)")
plt.legend()
plt.show()

# Mostramos coeficientes 
print("Pendiente: ", regr.coef_[0][0])
print("Intercepto: ", regr.intercept_[0])
print("R^2: ", regr.score(xG, scored))


