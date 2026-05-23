import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Cargar Datos
file_path = r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv"
df = pd.read_csv(file_path)

# print(f"Total de partidos: {len(df)}")

# 2. Preprocesamiento
# Convertimos variables categóricas a numéricas
# Resultado: w=2, d=1, l=0 (Arbitrario, pero sirve para clasificación)
map_result = {'w': 2, 'd': 1, 'l': 0}
df['target'] = df['result'].map(map_result)

# Factor Campo: h=1, a=0
map_ha = {'h': 1, 'a': 0}
df['is_home'] = df['h_a'].map(map_ha)

# Selección de Variables Predictoras (FEATURES)
# NO incluimos goles reales ni puntos, porque eso es lo que queremos predecir indirectamente.
features = ['xG', 'xGA', 'ppda_coef', 'oppda_coef', 'deep', 'deep_allowed', 'is_home']
X = df[features]
y = df['target']

# 3. División Entrenamiento / Test
# Usamos el 80% para entrenar y el 20% para examinar al modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Partidos de entrenamiento: {len(X_train)}")
print(f"Partidos de test: {len(X_test)}")

# 4. Crear y Entrenar el Modelo
# max_depth=3 para que el árbol sea legible en el gráfico (previene sobreajuste simple)
clf = DecisionTreeClassifier(random_state=42, max_depth=3, criterion='gini')
clf.fit(X_train, y_train)

# 5. Evaluación
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n--- Resultados del Árbol de Decisión ---")
print(f"Precisión (Accuracy): {acc:.2%}")
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Derrota', 'Empate', 'Victoria']))

# 6. Visualización del Árbol
plt.figure(figsize=(30, 12))
plot_tree(clf, 
        feature_names=features, 
        class_names=['Derrota', 'Empate', 'Victoria'], 
        filled=True, 
        rounded=True, 
        fontsize=8,
        precision=2)
plt.title(f"Árbol de Decisión (Profundidad 3) - Accuracy: {acc:.2%}")
plt.tight_layout()
plt.show()

# 7. Matriz de Confusión (Opcional, pero recomendada) (Entenderlo)
#cm = confusion_matrix(y_test, y_pred)
#plt.figure(figsize=(6, 5))
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['L', 'D', 'W'], yticklabels=['L', 'D', 'W'])
#plt.xlabel('Predicción')
#plt.ylabel('Realidad')
#plt.title('Matriz de Confusión')
#plt.show()
