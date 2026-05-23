import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree # OJO: Importamos Regressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')

# 1. CARGAR DATOS
# Asegúrate de que la ruta es correcta
file_path = r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv"
df = pd.read_csv(file_path)

# 2. DEFINIR LA VARIABLE OBJETIVO (TARGET)
# Queremos predecir la "Eficiencia" o "Suerte": (Goles Reales - Goles Esperados)
# - Valor Positivo (+): Overperformance (Marcó más de lo que debía).
# - Valor Negativo (-): Underperformance (Falló más de lo que debía).
# - Valor Cero (0): Rendimiento estándar.
df['efficiency_diff'] = df['scored'] - df['xG']

# 3. SELECCIÓN DE VARIABLES (FEATURES)
# Usamos las mismas métricas tácticas para ver si alguna influye en tener mejor puntería.
# Convertimos 'h_a' a número primero.
df['is_home'] = df['h_a'].map({'h': 1, 'a': 0})

features = ['xG', 'xGA', 'ppda_coef', 'oppda_coef', 'deep', 'deep_allowed', 'is_home']
X = df[features]
y = df['efficiency_diff'] # Ahora nuestra meta es este número continuo

# 4. DIVISIÓN TRAIN / TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. CREAR Y ENTRENAR EL MODELO (REGRESOR)
# Usamos max_depth=3 para que el árbol sea explicable en el TFG
# criterion='squared_error' es el estándar para regresión (minimizar el error cuadrático)
regressor = DecisionTreeRegressor(random_state=42, max_depth=3, criterion='squared_error')
regressor.fit(X_train, y_train)

# 6. EVALUACIÓN DEL MODELO
# En regresión no hay "Accuracy" (%). Hay "Cuánto me he equivocado" (Error).
y_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"\n--- Evaluación del Modelo de Regresión ---")
print(f"Error Medio Absoluto (MAE): {mae:.4f}")
print(f"Interpretación: De media, el modelo se equivoca por {mae:.2f} goles al predecir la diferencia.")

# 7. VISUALIZACIÓN DEL ÁRBOL
plt.figure(figsize=(35, 15)) # Lienzo ancho
plot_tree(regressor, 
          feature_names=features, 
          filled=True, 
          rounded=True, 
          fontsize=8, 
          precision=3) # Mostramos 3 decimales para ver bien las diferencias pequeñas
plt.title(f"Árbol de Regresión: Predicción de (Goles - xG)\nMAE: {mae:.3f}", fontsize=16)
plt.show()

# 8. IMPORTANCIA DE VARIABLES
# ¿Qué influye más en que un equipo marque más/menos de lo esperado?
importancia = pd.DataFrame({'Variable': features, 'Importancia': regressor.feature_importances_})
print("\n--- Variables más influyentes en la Eficiencia ---")
print(importancia.sort_values('Importancia', ascending=False))