import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # ¡Importamos el Bosque!
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuración visual
plt.style.use('seaborn-v0_8-whitegrid')

# --- PASO 1: Carga y Preparación ---
file_path = r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv"
df = pd.read_csv(file_path)

# Variable Objetivo: Eficiencia (Goles - xG)
df['efficiency_diff'] = df['scored'] - df['xG']

# Variables Predictoras (Features)
df['is_home'] = df['h_a'].map({'h': 1, 'a': 0})
features = ['xG', 'xGA', 'ppda_coef', 'oppda_coef', 'deep', 'deep_allowed', 'is_home']

X = df[features]
y = df['efficiency_diff']

# --- PASO 2: División de Datos ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- PASO 3: Entrenar el Random Forest ---
print("Entrenando el bosque... esto puede tardar unos segundos.")

# n_estimators=100: Creamos 100 árboles
# random_state=42: Para que el resultado sea reproducible
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) 
rf_model.fit(X_train, y_train)

# --- PASO 4: Predicción y Evaluación ---
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n--- RESULTADOS RANDOM FOREST ---")
print(f"Error Medio Absoluto (MAE): {mae:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print("Interpretación: El MAE nos dice cuánto nos equivocamos de media en cada partido.")

# --- PASO 5: Gráfico de Importancia de Variables ---
# Este es el gráfico MÁS IMPORTANTE para tu TFG
importancia = pd.DataFrame({
    'Variable': features,
    'Importancia': rf_model.feature_importances_
}).sort_values('Importancia', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Variable', data=importancia, palette='viridis')
plt.title('¿Qué variables deciden la Eficiencia? (Random Forest)')
plt.xlabel('Peso relativo (0 a 1)')
plt.show()

# --- PASO 6: Gráfico Realidad vs Predicción ---
# Para ver si el modelo acierta o no
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Línea perfecta
plt.xlabel('Eficiencia REAL (Goles - xG)')
plt.ylabel('Eficiencia PREDICHA por el Modelo')
plt.title('Dispersión de Predicciones (Línea Roja = Predicción Perfecta)')
plt.show()