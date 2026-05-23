import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb # ¡Importamos el rey del Machine Learning tabular!

# Configuración visual
plt.style.use('seaborn-v0_8-whitegrid')

# --- PASO 1: Carga y Preparación (Igual que en Random Forest) ---
file_path = r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv"
df = pd.read_csv(file_path)

df['efficiency_diff'] = df['scored'] - df['xG']
df['is_home'] = df['h_a'].map({'h': 1, 'a': 0})
features = ['xG', 'xGA', 'ppda_coef', 'oppda_coef', 'deep', 'deep_allowed', 'is_home']

X = df[features]
y = df['efficiency_diff']

# --- PASO 2: División de Datos ---
# Usamos el mismo random_state=42 para que la comparación sea 100% justa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- PASO 3: Entrenar el modelo XGBoost ---
print("Entrenando XGBoost... ajustando los árboles secuenciales.")

# A diferencia de Random Forest, aquí los árboles aprenden de los errores del anterior
xgb_model = xgb.XGBRegressor(
    n_estimators=100,      # Número de árboles
    learning_rate=0.1,     # Qué tan rápido aprende (0.1 es un buen estándar)
    max_depth=5,           # Profundidad máxima de cada árbol (evita el sobreajuste)
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# --- PASO 4: Predicción y Evaluación ---
y_pred_xgb = xgb_model.predict(X_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

print("\n--- RESULTADOS XGBOOST ---")
print(f"Error Medio Absoluto (MAE): {mae_xgb:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse_xgb:.4f}")

# --- PASO 5: Gráfico de Importancia de Variables ---
importancia_xgb = pd.DataFrame({
    'Variable': features,
    'Importancia': xgb_model.feature_importances_
}).sort_values('Importancia', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Variable', data=importancia_xgb, palette='magma') # Usamos otra paleta para diferenciarlo
plt.title('¿Qué variables deciden la Eficiencia? (XGBoost)')
plt.xlabel('Peso relativo (0 a 1)')
plt.show()

# --- PASO 6: Gráfico Realidad vs Predicción ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.3, color='green') # En verde para diferenciarlo del RF
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Eficiencia REAL (Goles - xG)')
plt.ylabel('Eficiencia PREDICHA por XGBoost')
plt.title('Dispersión de Predicciones XGBoost (Línea Roja = Predicción Perfecta)')
plt.show()