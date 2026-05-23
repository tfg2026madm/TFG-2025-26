import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')

# Cargar datos
file_path = r"C:\Users\pablo\OneDrive - Universidad Complutense de Madrid (UCM)\5º (1º CUATRI)\TFG\AnálisisPythonTFG\understat_per_game.csv"
df = pd.read_csv(file_path)

# --- PASO 1: Separar Datos Local (h) y Visitante (a) ---
df_home = df[df['h_a'] == 'h']
df_away = df[df['h_a'] == 'a']

# Función para realizar regresión y mostrar resultados
def analizar_regresion(data, label, color, ax):
    X = data[['xG']]
    y = data[['scored']]
    
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    
    m = regr.coef_[0][0]
    b = regr.intercept_[0]
    r2 = regr.score(X, y)
    
    print(f"--- {label} ---")
    print(f"Pendiente (Eficiencia): {m:.4f}")
    print(f"R^2 (Explicabilidad):   {r2:.4f}")
    
    # Visualización
    ax.scatter(data['xG'], data['scored'], alpha=0.1, color=color, label=f'Partidos ({label})')
    
    # Línea de regresión
    x_line = np.linspace(0, data['xG'].max(), 100).reshape(-1, 1)
    y_line = regr.predict(x_line)
    ax.plot(x_line, y_line, color=color, linewidth=2, linestyle='--', label=f'Regresión {label} (m={m:.2f})')
    
    return m, r2

# --- PASO 2: Ejecutar Análisis ---
fig, ax = plt.subplots(figsize=(10, 8))

print("RESULTADOS COMPARATIVOS:")
m_home, r2_home = analizar_regresion(df_home, "Local", "blue", ax)
m_away, r2_away = analizar_regresion(df_away, "Visitante", "red", ax)

# --- PASO 3: Configuración del Gráfico ---
ax.set_title("Comparativa de Eficiencia: Local vs Visitante", fontsize=16)
ax.set_xlabel("xG (Goles Esperados)")
ax.set_ylabel("Goles Reales")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

# Añadir texto explicativo en el gráfico
plt.text(0.05, 0.95, f"Local: m={m_home:.3f}, R2={r2_home:.3f}", transform=ax.transAxes, color='blue', fontweight='bold')
plt.text(0.05, 0.90, f"Visitante: m={m_away:.3f}, R2={r2_away:.3f}", transform=ax.transAxes, color='red', fontweight='bold')

plt.tight_layout()
plt.show()
