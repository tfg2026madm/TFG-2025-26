import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Creamos el dataFrame con Pendiente, Intercepto y Coeficiente
data = {
    "Liga": ["Bundesliga", "EPL", "LaLiga", "Ligue 1", "RFPL", "Serie A"],
    "Pendiente": [0.9639794734950106, 0.9438123630421296, 0.9579190119379731, 0.9805463697053016, 0.9933557872205834, 0.9080248089744112],
    "Intercepto": [0.08875718303999247, 0.09643649056714687, 0.06658507691028404, 0.06847324244694164, 0.048676514890171996, 0.1624246001007259],
    "R^2": [0.43164993503213944, 0.40283463194659985, 0.4413621652181777, 0.41694069115852883, 0.4111000530721236, 0.37849523498105264],
}

df = pd.DataFrame(data)

fig, axs = plt.subplots(3, 1, figsize=(8, 12))

axs[0].bar(df["Liga"], df["Pendiente"], width=0.5)
axs[0].set_ylabel("Pendiente")

axs[1].bar(df["Liga"], df["Intercepto"], width=0.5)
axs[1].set_ylabel("Intercepto")

axs[2].bar(df["Liga"], df["R^2"], width=0.5)
axs[2].set_ylabel("R^2")

plt.xticks(rotation=15)   # <-- INCLINAR ETIQUETAS
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)   # <-- MÁS SEPARACIÓN ENTRE GRÁFICAS
plt.show()