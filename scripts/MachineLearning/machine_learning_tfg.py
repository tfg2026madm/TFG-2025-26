import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# =========================================================
# 1. CONFIGURACIÓN GENERAL
# =========================================================

RUTA_DATOS = "datos_TFG_xG.xlsx"
CARPETA_SALIDA = "graficas_tfg"

os.makedirs(CARPETA_SALIDA, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


# =========================================================
# 2. CARGA DEL DATASET
# =========================================================

df = pd.read_excel(RUTA_DATOS)


# =========================================================
# 3. VARIABLE OBJETIVO
# =========================================================

df["diff_xG"] = df["scored"] - df["xG"]


# =========================================================
# 4. VARIABLE EXTRA: is_home
# =========================================================

if "h_a" in df.columns:
    df["is_home"] = df["h_a"].astype(str).str.lower().isin(
        ["home", "h", "local", "casa"]
    ).astype(int)

elif "home/away" in df.columns:
    df["is_home"] = df["home/away"].astype(str).str.lower().isin(
        ["home", "h", "local", "casa"]
    ).astype(int)

elif "home_away" in df.columns:
    df["is_home"] = df["home_away"].astype(str).str.lower().isin(
        ["home", "h", "local", "casa"]
    ).astype(int)

else:
    df["is_home"] = 0


# =========================================================
# 5. VARIABLES PREDICTORAS
# =========================================================

features = [
    "xG",
    "xGA",
    "deep",
    "deep_allowed",
    "ppda_coef",
    "oppda_coef",
    "is_home"
]

target = "diff_xG"

X = df[features].copy()
y = df[target].copy()


# =========================================================
# 6. DIVISIÓN DE DATOS
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================================================
# 7. MODELOS
# =========================================================

tree = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

xgb = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    objective="reg:squarederror",
    eval_metric="rmse"
)


# =========================================================
# 8. ENTRENAMIENTO
# =========================================================

tree.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)


# =========================================================
# 9. PREDICCIONES
# =========================================================

pred_tree = tree.predict(X_test)
pred_rf = rf.predict(X_test)
pred_xgb = xgb.predict(X_test)


# =========================================================
# 10. MÉTRICAS
# =========================================================

def evaluar_modelo(nombre, y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)

    return {
        "Modelo": nombre,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


resultados = [
    evaluar_modelo("Árbol de Decisión", y_test, pred_tree),
    evaluar_modelo("Random Forest", y_test, pred_rf),
    evaluar_modelo("XGBoost", y_test, pred_xgb)
]

resultados_df = pd.DataFrame(resultados)

print("\n================ RESULTADOS =================")
print(resultados_df.round(4))
print("============================================")

resultados_df.round(4).to_csv(
    os.path.join(CARPETA_SALIDA, "tabla_metricas_modelos.csv"),
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 11. FIGURA 1: ÁRBOL DE REGRESIÓN
# =========================================================

mae_tree = mean_absolute_error(y_test, pred_tree)

fig, ax = plt.subplots(figsize=(18, 10))

plot_tree(
    tree,
    feature_names=features,
    filled=True,
    rounded=True,
    fontsize=10,
    impurity=True,
    proportion=False,
    precision=3,
    ax=ax
)

plt.tight_layout()
plt.savefig(
    os.path.join(CARPETA_SALIDA, "arbol_regresion.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# =========================================================
# 12. FIGURA 2: IMPORTANCIA RELATIVA - RANDOM FOREST
# =========================================================

importancias_rf = pd.DataFrame({
    "Variable": features,
    "Importancia": rf.feature_importances_
}).sort_values(by="Importancia", ascending=False)

colors_rf = plt.cm.viridis(np.linspace(0.12, 0.88, len(importancias_rf)))

fig, ax = plt.subplots(figsize=(13, 6.2))

ax.barh(
    importancias_rf["Variable"],
    importancias_rf["Importancia"],
    color=colors_rf
)

ax.invert_yaxis()

ax.set_xlabel("Peso relativo (0 a 1)", fontsize=12)
ax.set_ylabel("Variable", fontsize=12)

ax.grid(axis="x", alpha=0.7)
ax.grid(axis="y", visible=False)

plt.tight_layout()
plt.savefig(
    os.path.join(CARPETA_SALIDA, "importancia_random_forest.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# =========================================================
# 13. FIGURA 3: IMPORTANCIA RELATIVA - XGBOOST
# =========================================================

importancias_xgb = pd.DataFrame({
    "Variable": features,
    "Importancia": xgb.feature_importances_
})

importancias_xgb = importancias_xgb[importancias_xgb["Variable"] != "is_home"]
importancias_xgb = importancias_xgb.sort_values(by="Importancia", ascending=False)

colors_xgb = plt.cm.viridis(np.linspace(0.9, 0.2, len(importancias_xgb)))

fig, ax = plt.subplots(figsize=(17, 8.5))

ax.barh(
    importancias_xgb["Variable"],
    importancias_xgb["Importancia"],
    color=colors_xgb
)

ax.invert_yaxis()

ax.set_xlabel("Peso relativo (0 a 1)", fontsize=17)
ax.set_ylabel("Variable", fontsize=17)

ax.tick_params(axis="both", labelsize=13)
ax.grid(axis="x", alpha=0.65)
ax.grid(axis="y", visible=False)

plt.tight_layout()
plt.savefig(
    os.path.join(CARPETA_SALIDA, "importancia_xgboost.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# =========================================================
# 14. FIGURA 4: DISPERSIÓN DE PREDICCIONES
# =========================================================

fig, ax = plt.subplots(figsize=(13, 8))

ax.scatter(
    y_test,
    pred_xgb,
    color="blue",
    alpha=0.35,
    s=38
)

valor_min = min(y_test.min(), pred_xgb.min())
valor_max = max(y_test.max(), pred_xgb.max())

ax.plot(
    [valor_min, valor_max],
    [valor_min, valor_max],
    color="red",
    linestyle="--",
    linewidth=2
)

ax.set_title("(Línea Roja = Predicción Perfecta)", fontsize=17)
ax.set_xlabel("Eficiencia REAL (Goles - xG)", fontsize=13)
ax.set_ylabel("Eficiencia PREDICHA por el Modelo", fontsize=13)

ax.grid(True, alpha=0.7)

plt.tight_layout()
plt.savefig(
    os.path.join(CARPETA_SALIDA, "dispersion_predicciones.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()