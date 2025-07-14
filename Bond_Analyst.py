import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

bonos = pd.read_excel("bonos.xlsx")

# Limpiar nombres de columnas (quita espacios al inicio y final)
bonos.columns = bonos.columns.str.strip()

# Hacer una copia real del Yield original antes de cualquier conversión o escalado
bonos["Yield_original"] = bonos["Yield"].copy()

# Ahora sí, convierte a numérico y escala
cols_to_numeric = ["1 Week", "1 Month", "1 Year", "3 Years", "Yield"]
for col in cols_to_numeric:
    bonos[col] = pd.to_numeric(bonos[col], errors='coerce')

# Mapeo de categorías a números
categoria_map = {
    "Strong Sell": 0,
    "Sell": 1,
    "Neutral": 2,
    "Buy": 3,
    "Strong Buy": 5
}

# Convertir columnas 'Weekly' y 'Monthly' a numéricas
bonos["Weekly_num"] = bonos["Weekly"].map(categoria_map)
bonos["Monthly_num"] = bonos["Monthly"].map(categoria_map)

# Asegurar que las columnas usadas sean numéricas ANTES de escalar
cols_to_numeric = ["1 Week", "1 Month", "1 Year", "3 Years"]
for col in cols_to_numeric:
    bonos[col] = pd.to_numeric(bonos[col], errors='coerce')

# Aplicar RobustScaler solo a "3 Years" y "1 Year"
robust_scaler = RobustScaler()
bonos[["3 Years", "1 Year"]] = robust_scaler.fit_transform(bonos[["3 Years", "1 Year"]])

# Re-escalar ambas columnas a [0, 1] usando MinMaxScaler
minmax_scaler = MinMaxScaler()
bonos[["3 Years", "1 Year"]] = minmax_scaler.fit_transform(bonos[["3 Years", "1 Year"]])

# Seleccionar solo las columnas numéricas para escalar (excepto "3 Years" y "1 Year" ya escaladas)
num_cols = bonos.select_dtypes(include=["number"]).columns.drop(["3 Years", "1 Year"])

# Estandarizar el resto con MinMaxScaler
scaler = MinMaxScaler()
bonos[num_cols] = scaler.fit_transform(bonos[num_cols])

# Crear calidad fundamental y añadirla al DataFrame
bonos["Calidad_Fundamental"] = (
    bonos["Rating S&P"]**1.5 +
    bonos["Yield"]**1.2 -
    (0.4* bonos["Maturity"] + 0.3 * bonos["Convexity"] + 0.3 * bonos["MD"])
)

# Crear la métrica técnica y añadirla al DataFrame
bonos["Tecnico"] = (
    ((bonos["1 Week"] + bonos["1 Month"]) / 2) * 
    (0.6 * bonos["Weekly_num"] + 0.4 * bonos["Monthly_num"]) + 
    0.2 * bonos["1 Year"] - 
    0.1 * bonos["3 Years"]
)

# Crear la métrica macro y añadirla al DataFrame (versión avanzada)
bonos["Macro_Score"] = (
    # Efectos directos
    -0.30 * bonos["Interest Rate"] +
    -0.20 * bonos["Unemployment Rate"] +
     0.35 * bonos["GDP YoY"] +
    -0.20 * bonos["CPI YoY"] +
     0.25 * bonos["Trade Balance"] +
    -0.30 * bonos["Deuda/PIB actual"] +
    -0.30 * bonos["Déficit fiscal (2024‑25)"] +

    # Interacciones macroeconómicas clave
    -0.15 * bonos["Interest Rate"] * bonos["Déficit fiscal (2024‑25)"] +
    -0.10 * bonos["CPI YoY"] * (1 / (bonos["GDP YoY"] + 0.1)) +
    -0.10 * bonos["Deuda/PIB actual"] * (1 / (bonos["GDP YoY"] + 0.1)) +
    -0.05 * bonos["Unemployment Rate"] * bonos["Déficit fiscal (2024‑25)"]
)

# Crear la métrica de estimaciones de deuda y añadirla al DataFrame
bonos["Estimaciones_Deuda"] = (
    -0.4 * bonos["Deuda/PIB +1 año"] +
    -0.3 * bonos["Deuda/PIB +5 años"] +
    -0.3 * bonos["Deuda/PIB +10 años"] +

    # Penalización por crecimiento acelerado de la deuda
    -0.2 * (bonos["Deuda/PIB +5 años"] - bonos["Deuda/PIB +1 año"]) +
    -0.2 * (bonos["Deuda/PIB +10 años"] - bonos["Deuda/PIB +5 años"]) +

    # Penalización adicional si la deuda se acelera con el tiempo
    -0.2 * ((bonos["Deuda/PIB +10 años"] - bonos["Deuda/PIB +5 años"]) - 
            (bonos["Deuda/PIB +5 años"] - bonos["Deuda/PIB +1 año"]))
)

# Estandarizar las 4 métricas en min-max
metricas = ["Calidad_Fundamental", "Tecnico", "Macro_Score", "Estimaciones_Deuda"]
scaler_metricas = MinMaxScaler()
bonos[[m + "_std" for m in metricas]] = scaler_metricas.fit_transform(bonos[metricas])

# Crear una métrica compuesta ponderada con las 4 métricas estandarizadas
bonos["Score_Final"] = (
    0.5 * bonos["Calidad_Fundamental_std"] +
    0.05 * bonos["Tecnico_std"] +
    0.4 * bonos["Macro_Score_std"] +
    0.05 * bonos["Estimaciones_Deuda_std"]
)

# Guardar el DataFrame estandarizado a un nuevo archivo Excel
bonos.to_excel("bonos_estandarizados.xlsx", index=False)

# Regresión lineal entre Score_Final y Yield (sin estandarizar)
# Eliminar filas con NaN en Score_Final, Yield o Country antes de la regresión
reg_data = bonos[["Score_Final", "Yield_original", "Country"]].dropna()
X = reg_data[["Score_Final"]]
y = reg_data["Yield_original"]

reg = LinearRegression()
reg.fit(X, y)

print(f"Coeficiente: {reg.coef_[0]:.4f}")
print(f"Intercepto: {reg.intercept_:.4f}")
print(f"R^2: {reg.score(X, y):.4f}")

# Gráfico de dispersión y recta de regresión, coloreando por país
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(data=reg_data, x="Score_Final", y="Yield_original", hue="Country", palette="tab10", alpha=0.7)
plt.plot(X, reg.predict(X), color="red", label="Regresión")
plt.xlabel("Score_Final")
plt.ylabel("Yield (sin estandarizar)")
plt.title("Regresión lineal: Score_Final vs Yield (coloreado por Country)")
plt.legend()
plt.show()

# Regresión cuadrática entre Score_Final y Yield (sin estandarizar)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(reg_data[["Score_Final"]])

reg_quad = LinearRegression()
reg_quad.fit(X_poly, y)

print("\nRegresión cuadrática:")
print(f"Coeficientes: {reg_quad.coef_}")
print(f"Intercepto: {reg_quad.intercept_:.4f}")
print(f"R^2: {reg_quad.score(X_poly, y):.4f}")

# Gráfico de dispersión y curva de regresión cuadrática, coloreando por país
import numpy as np

plt.figure(figsize=(8, 6))
sns.scatterplot(data=reg_data, x="Score_Final", y="Yield_original", hue="Country", palette="tab10", alpha=0.7)

# Para la curva cuadrática, ordena los valores de X para una línea suave
x_sorted = np.sort(reg_data["Score_Final"])
x_sorted_poly = poly.transform(x_sorted.reshape(-1, 1))
y_pred_quad = reg_quad.predict(x_sorted_poly)
plt.plot(x_sorted, y_pred_quad, color="green", label="Regresión cuadrática")

plt.xlabel("Score_Final")
plt.ylabel("Yield (sin estandarizar)")
plt.title("Regresión cuadrática: Score_Final vs Yield (coloreado por Country)")
plt.legend()
plt.show()

# Predicciones de ambas regresiones
reg_data["Yield_pred_lineal"] = reg.predict(X)
reg_data["Yield_pred_cuadratica"] = reg_quad.predict(X_poly)

# Observaciones donde el valor real (no estandarizado) es mayor al predicho en al menos una regresión
# (Yield debe ser la variable original, no estandarizada)
mayor_predicho = reg_data[
    (reg_data["Yield_original"] > reg_data["Yield_pred_lineal"]) | 
    (reg_data["Yield_original"] > reg_data["Yield_pred_cuadratica"])
]

# Añadir la columna 'Name' si existe en el DataFrame original
if "Name" in bonos.columns:
    mayor_predicho = mayor_predicho.merge(
        bonos[["Name"]], left_index=True, right_index=True, how="left"
    )

# Observaciones donde el valor real es mayor al predicho en la lineal
mayor_lineal = reg_data[reg_data["Yield_original"] > reg_data["Yield_pred_lineal"]]
if "Name" in bonos.columns:
    mayor_lineal = mayor_lineal.merge(
        bonos[["Name"]], left_index=True, right_index=True, how="left"
    )
cols_export = ["Name", "Score_Final", "Yield_original", "Yield_pred_lineal", "Country"]
cols_export = [col for col in cols_export if col in mayor_lineal.columns]
mayor_lineal[cols_export].to_excel("mayor_que_predicho_lineal.xlsx", index=False)

# Observaciones donde el valor real es mayor al predicho en la cuadrática
mayor_cuadratica = reg_data[reg_data["Yield_original"] > reg_data["Yield_pred_cuadratica"]]
if "Name" in bonos.columns:
    mayor_cuadratica = mayor_cuadratica.merge(
        bonos[["Name"]], left_index=True, right_index=True, how="left"
    )
cols_export = ["Name", "Score_Final", "Yield_original", "Yield_pred_cuadratica", "Country"]
cols_export = [col for col in cols_export if col in mayor_cuadratica.columns]
mayor_cuadratica[cols_export].to_excel("mayor_que_predicho_cuadratica.xlsx", index=False)


