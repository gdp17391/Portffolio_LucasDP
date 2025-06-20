import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from linearmodels.panel import PanelOLS
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import skew, kurtosis, shapiro, anderson
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_ljungbox
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Cargar los archivos Excel
fundamentales = pd.read_excel('fundamentales tfg.xlsx')
var_control = pd.read_excel('var control tfg.xlsx')

# Convertir TICKER y DATE a categórico
fundamentales['TICKER'] = pd.Categorical(fundamentales['TICKER'])
fundamentales['DATE'] = pd.Categorical(fundamentales['DATE'])
var_control['DATE'] = pd.Categorical(var_control['DATE'])

# Fusionar los DataFrames en DATE
data = pd.merge(fundamentales, var_control, on='DATE', how='left')

# Eliminar espacios adicionales en los nombres de las columnas
data.columns = data.columns.str.strip()

# Fix DATE data type after merge
data['DATE'] = pd.Categorical(data['DATE'])

# Sort the data by TICKER and DATE to ensure correct order for return calculation
data = data.sort_values(['TICKER', 'DATE'])

# Calcular el crecimiento de PRICE entre trimestres para cada empresa y aplicar shift(-1)
data['PRICE_Growth'] = data.groupby('TICKER', observed=True)['PRICE'].pct_change()

# Calcular la correlación entre todas las variables y PRICE_Growth antes de estandarizarlas
correlation_before_scaling = data.corr(numeric_only=True)['PRICE_Growth'].dropna()

# Mostrar la correlación de cada variable con PRICE_Growth
print("Correlación con PRICE_Growth antes de estandarizar:")
print(correlation_before_scaling)

# Exportar la correlación a un archivo Excel
correlation_before_scaling.to_frame(name='Correlation').to_excel('correlation_with_price_growth_before_scaling.xlsx')

print("Columnas disponibles en el DataFrame:")
print(data.columns)

# Lista de variables a estandarizar
variables_a_estandarizar = ['CASH', 'DEBT', 'EPS', 'FCFMG', 'NETPROFITMG', 
                            'OPERATINGMG', 'QUICK', 'ROA', 'ROE', 'EMPLOYEES', 'REVENUES']

# Aplicar Min-Max Scaling para cada trimestre (DATE)
scaler = MinMaxScaler()
data[variables_a_estandarizar] = data.groupby('DATE', observed=True)[variables_a_estandarizar].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
)

# Excluir Q1_2025 después de calcular PRICE_Growth y estandarizar
data = data[data['DATE'] != 'Q1_2025']

data[variables_a_estandarizar] = data.groupby('TICKER', observed=True)[variables_a_estandarizar].shift(1)

# Definir los pesos iniciales para cada variable
weights = {
    'CASH': 0.2,
    'DEBT': -1,
    'EPS': 0.2,
    'FCFMG': 0.2,
    'NETPROFITMG': 0.25,
    'OPERATINGMG': 0.25,
    'QUICK': 0.2,
    'ROA': 0.15,
    'ROE': 0.15,
    'EMPLOYEES': 0.2,
    'REVENUES': 0.2
}

# Verificar que los pesos sumen 1
if not np.isclose(sum(weights.values()), 1):
    raise ValueError("Los pesos no suman 1. Por favor, ajusta los valores.")

# Filtrar los pesos iniciales para que coincidan con las columnas en variables_a_estandarizar
filtered_weights_initial = {key: weights[key] for key in variables_a_estandarizar if key in weights}

# Calcular el índice inicial para cada empresa y trimestre
data['INDEX'] = data[filtered_weights_initial.keys()].dot(pd.Series(filtered_weights_initial))

# Eliminar espacios adicionales en los nombres de las columnas
data.columns = data.columns.str.strip()

# Definir los grupos de variables
MACRO_VARS = ['Interest rates', 'Real GDP Growth', 'Exchange Rates', 
              'Credit spreads', 'Unemployment rate']

INDICATOR_VARS = ['VIX', 'XLV', 'SPY']

FEATURE_VARS = ['VOLUME', 'SHARES']

# Filtrar las columnas de cada grupo que están presentes en el DataFrame
MACRO_VARS = [var for var in MACRO_VARS if var in data.columns]
INDICATOR_VARS = [var for var in INDICATOR_VARS if var in data.columns]
FEATURE_VARS = [var for var in FEATURE_VARS if var in data.columns]

# Función para calcular log-differences
def log_difference(series):
    # Manejar valores negativos sumando un desplazamiento positivo
    min_value = series.min()
    if (min_value <= 0):
        series += abs(min_value) + 1e-6  # Desplazar para que todos los valores sean positivos
    return np.log(series).diff()

# Estandarizar MACRO_VARS usando log-differences, excepto 'Real GDP Growth' y 'Interest rates'
if MACRO_VARS:
    for var in MACRO_VARS:
        if var == 'Interest rates':
            # Calcular el cambio porcentual para Interest rates
            data[var] = data.groupby('TICKER', observed=True)[var].pct_change()
        elif var != 'Real GDP Growth':  # Excluir 'Real GDP Growth' del cálculo de log-differences
            # Calcular log-differences para las demás variables
            data[var] = data.groupby('TICKER', observed=True)[var].transform(log_difference)
else:
    print("No se encontraron columnas para MACRO_VARS en el DataFrame.")

# Estandarizar INDICATOR_VARS
if INDICATOR_VARS:
    for var in INDICATOR_VARS:
        data[var] = log_difference(data[var])
else:
    print("No se encontraron columnas para INDICATOR_VARS en el DataFrame.")

# Estandarizar FEATURE_VARS (incluyendo SHARES) usando log-differences dentro de cada empresa (por TICKER)
if FEATURE_VARS:
    for var in FEATURE_VARS:
        if var == 'SHARES':
            # Calcular log-differences para SHARES
            data['SHARES'] = data.groupby('TICKER', observed=True)['SHARES'].transform(log_difference)
            
            # Aplicar Min-Max Scaling por período (DATE) después de log-differences
            scaler_shares = MinMaxScaler()
            data['SHARES'] = data.groupby('DATE', observed=True)['SHARES'].transform(
                lambda x: scaler_shares.fit_transform(x.values.reshape(-1, 1)).flatten()
            )
        else:
            # Estandarizar las demás variables de FEATURE_VARS usando log-differences
            data[var] = data.groupby('TICKER', observed=True)[var].transform(log_difference)
else:
    print("No se encontraron columnas para FEATURE_VARS en el DataFrame.")

# Asegurarse de que DATE no tenga espacios adicionales
data['DATE'] = data['DATE'].astype(str).str.strip()

# Definir las variables independientes (X) y la variable dependiente (y)
X_vars = ['INDEX'] + MACRO_VARS + INDICATOR_VARS + FEATURE_VARS
y_var = 'PRICE_Growth'

# Verificar que las variables estén en el DataFrame
if not X_vars:
    raise ValueError("No se encontraron variables independientes en el DataFrame.")
if y_var not in data.columns:
    raise ValueError(f"La variable dependiente '{y_var}' no se encuentra en el DataFrame.")

# Eliminar filas con NaN en las variables seleccionadas
data = data.dropna(subset=X_vars + [y_var])

# Definir X (independientes) y y (dependiente)
X = data[X_vars]
y = data[y_var]

# Calcular la correlación entre las variables independientes y PRICE_Growth
correlation_data = data[X_vars + [y_var]].corr()[y_var].drop(y_var)

# Mostrar la correlación de cada variable con PRICE_Growth
print("Correlación con PRICE_Growth:")
print(correlation_data)

# Exportar la correlación a un archivo Excel
correlation_data.to_frame(name='Correlation').to_excel('correlation_with_price_growth.xlsx')

# Dividir los datos en conjuntos de entrenamiento y prueba
X = data[X_vars]
y = data[y_var]

# Calcular el Variance Inflation Factor (VIF) para las variables independientes
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Iteratively eliminate variables with VIF > 10
while True:
    vif_data = calculate_vif(X)
    print("Variance Inflation Factor (VIF):")
    print(vif_data)

    # Export the VIF data to an Excel file
    vif_data.to_excel('vif_data.xlsx', index=False)

    # Identify variables with VIF > 10
    high_vif_features = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
    if not high_vif_features:
        break  # Exit the loop if no variables have VIF > 10

    print(f"Removing variables with VIF > 10: {high_vif_features}")
    X = X.drop(columns=high_vif_features)

# Asegurarse de que TICKER y DATE sean índices para el modelo de panel
data = data.set_index(['TICKER', 'DATE'])

# Verificar que las variables estén en el DataFrame
if not X_vars:
    raise ValueError("No se encontraron variables independientes en el DataFrame.")
if y_var not in data.columns:
    raise ValueError(f"La variable dependiente '{y_var}' no se encuentra en el DataFrame.")

# Eliminar filas con NaN en las variables seleccionadas
data = data.dropna(subset=X_vars + [y_var])

# Calcular el Variance Inflation Factor (VIF) para las variables independientes
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Dividir los datos en conjuntos de entrenamiento y prueba después de eliminar variables con alto VIF
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar Lasso with the remaining variables
lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)

# Mostrar los coeficientes de las variables seleccionadas
selected_features = pd.DataFrame({'Feature': X.columns, 'Coefficient': lasso.coef_})
print("Coeficientes de Lasso:")
print(selected_features)

# Filtrar las variables con coeficientes diferentes de 0
selected_features = selected_features[selected_features['Coefficient'] != 0]
selected_vars = selected_features['Feature'].tolist()
print(f"Variables seleccionadas después de Lasso: {selected_vars}")

# Redefinir X con las variables seleccionadas
X = X[selected_vars]

# Agregar una constante para el término independiente
X = sm.add_constant(X)

# Ajustar el modelo OLS
model = sm.OLS(y, X).fit(cov_type='HC3')  # Usar errores estándar robustos para heterocedasticidad

# Mostrar los resultados de la regresión
print(model.summary())

# Calcular el Mean Squared Error (MSE)
mse = np.mean((model.resid) ** 2)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Obtener los coeficientes, R² y R² ajustado
coefficients = model.params
r_squared = model.rsquared
adjusted_r_squared = model.rsquared_adj

print("\nCoefficients:")
print(coefficients)
print(f"\nR²: {r_squared:.4f}")
print(f"Adjusted R²: {adjusted_r_squared:.4f}")

# Exportar los coeficientes de Lasso a un archivo Excel
selected_features.to_excel('lasso_coefficients.xlsx', index=False)

# Exportar los resultados de la regresión OLS a un archivo Excel
coefficients.to_excel('ols_coefficients.xlsx', index=True)

# Exportar el índice inicial a un archivo Excel
data.reset_index(inplace=True)
data[['TICKER', 'DATE', 'INDEX']].to_excel('initial_index.xlsx', index=False)

# Parámetro alpha para ponderar MSE y correlación
alpha = 0.25  # Ajusta este valor según la importancia relativa de mse y correlation

data = data.dropna(subset=variables_a_estandarizar + [y_var])  # Eliminar filas con NaN
y = data[y_var]  # Actualizar y con los datos filtrados

def objective_function(**new_weights):
    # Verificar que los pesos estén dentro de los rangos (para depuración)
    for key, value in new_weights.items():
        if key == 'DEBT':
            if not (-0.99 <= value <= -0.01):
                print(f"Advertencia: {key}={value} fuera del rango [-0.99, -0.01], ajustando...")
                new_weights[key] = max(-0.99, min(-0.01, value))
        else:
            if not (0.01 <= value <= 0.99):
                print(f"Advertencia: {key}={value} fuera del rango [0.01, 0.99], ajustando...")
                new_weights[key] = max(0.01, min(0.99, value))

    # Calcular la suma total de los pesos
    total_weight = sum(new_weights.values())
    if total_weight == 0:  # Evitar división por cero
        print("Advertencia: Suma de pesos es 0, devolviendo valor muy negativo")
        return -1e10  # Valor muy negativo en lugar de -inf
    
    # Normalizar los pesos para que sumen 1
    normalized_weights = {key: value / total_weight for key, value in new_weights.items()}
    
    # Calcular el índice con los nuevos pesos
    data['INDEX2'] = data[variables_a_estandarizar].dot(pd.Series(normalized_weights))
    
    # Actualizar X con el nuevo INDEX2
    X_vars_adjusted = [var for var in selected_vars if var != 'INDEX'] + ['INDEX2']
    X_new = sm.add_constant(data[X_vars_adjusted])
    
    # Ajustar el modelo OLS con el nuevo X_new
    try:
        model = sm.OLS(y, X_new).fit(cov_type='HC3')
        mse = mean_squared_error(y, model.predict(X_new))
    except Exception as e:
        print(f"Error en OLS: {e}, devolviendo valor muy negativo")
        return -1e10  # Valor muy negativo en lugar de -inf

    # Calcular la correlación entre 'INDEX2' y 'PRICE_Growth'
    correlation = data[['INDEX2', 'PRICE_Growth']].corr().iloc[0, 1]
    if pd.isna(correlation) or np.isinf(correlation):
        print(f"Correlación inválida: {correlation}, devolviendo valor muy negativo")
        return -1e10  # Valor muy negativo en lugar de -inf

    # Mostrar los valores calculados en esta iteración
    print(f"MSE: {mse:.7f}")
    print(f"Correlación entre INDEX2 y PRICE_Growth: {correlation:.4f}")

    # Estandarizar el MSE a una escala de [0, 1]
    mse_score = 1 / (1 + mse)
    if np.isinf(mse_score) or np.isnan(mse_score):
        print(f"MSE_score inválido: {mse_score}, devolviendo valor muy negativo")
        return -1e10*9999  # Valor muy negativo en lugar de -inf

    # Penalizar pesos negativos en fundamentales que deberían ser positivos
    positive_vars = ['ROE', 'EPS', 'NETPROFITMG', 'OPERATINGMG', 'QUICK', 'CASH', 'REVENUES']
    penalty = sum(max(0, -normalized_weights.get(key, 0)) for key in positive_vars if key in normalized_weights) * 0.5
    print(f"Penalización por pesos negativos: {penalty:.4f}")

    # Regularización L2 para los pesos
    l2_penalty = 0.01 * sum(w ** 2 for w in normalized_weights.values())
    print(f"Penalización L2: {l2_penalty:.4f}")

    # Calcular la métrica a optimizar
    metric = alpha * mse_score + (1 - alpha) * correlation - penalty - l2_penalty  
    return metric       

# Crear los límites para cada peso
pbounds = {
    key: (-0.99, -0.01) if key == 'DEBT' else (0.01, 0.99)
    for key in weights.keys()
}

# Inicializar la optimización bayesiana
from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds,
    verbose=2,
    random_state=42,
)

# Definir los pesos iniciales como una copia de weights*
initial_weights = weights.copy()

# Establecer los pesos iniciales como punto de partida
optimizer.probe(params=initial_weights, lazy=True)

# Ejecutar la optimización
optimizer.maximize(init_points=40, n_iter=400)

# Obtener los mejores pesos
best_weights = optimizer.max['params']

# Normalizar los mejores pesos para que sumen 1 y respeten los límites de -1 a 1
total_weight = sum(best_weights.values())
if total_weight != 0:  # Evitar división por cero
    best_weights = {key: max(-1, min(1, value / total_weight)) for key, value in best_weights.items()}
else:
    print("Suma de pesos optimizados es 0, no se puede normalizar")
    best_weights = weights  # Usar los pesos iniciales como respaldo

print("Mejores pesos encontrados:", best_weights)

# Filtrar los pesos optimizados para que coincidan con las columnas en variables_a_estandarizar
filtered_weights = {key: best_weights[key] for key in variables_a_estandarizar if key in best_weights}

# Recalcular el índice con los pesos filtrados
data['INDEX2'] = data[filtered_weights.keys()].dot(pd.Series(filtered_weights))

# Exportar el índice optimizado a un archivo Excel
data.reset_index(inplace=True)
data[['TICKER', 'DATE', 'INDEX2', y_var]].to_excel('optimized_index.xlsx', index=False)

# Recalcular la correlación y el MSE con el nuevo INDEX2
X_vars_adjusted = [var for var in selected_vars if var != 'INDEX'] + ['INDEX2']
X_new = sm.add_constant(data[X_vars_adjusted])

model = sm.OLS(y, X_new).fit(cov_type='HC3')
optimized_correlation = data[['INDEX2', 'PRICE_Growth']].corr().iloc[0, 1]
final_mse = mean_squared_error(y, model.predict(X_new))

print(model.summary())
print(f"\nCorrelación optimizada entre INDEX2 y PRICE_Growth: {optimized_correlation:.6f}")
print(f"Mean Squared Error (MSE) después de la optimización: {final_mse:.8f}")

# Obtener los coeficientes, R² y R² ajustado
coefficients = model.params
r_squared = model.rsquared
adjusted_r_squared = model.rsquared_adj

print("\nCoefficients:")
print(coefficients)
print(f"\nR²: {r_squared:.4f}")
print(f"Adjusted R²: {adjusted_r_squared:.4f}")

# Validación Cruzada
model_sk = LinearRegression().fit(X_new, y)
cv_scores = cross_val_score(model_sk, X_new, y, cv=5, scoring='neg_mean_squared_error')
cv_mse_mean = -cv_scores.mean()
cv_mse_std = cv_scores.std()

print(f"\nCross-Validation Metrics:")
print(f"Cross-Validated MSE (Mean): {cv_mse_mean:.7f}")
print(f"Cross-Validated MSE (Std Dev): {cv_mse_std:.7f}")

# Predicción Fuera de Muestra
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
model_train = sm.OLS(y_train, sm.add_constant(X_train)).fit(cov_type='HC3')
y_pred_test = model_train.predict(sm.add_constant(X_test))
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"\nTest Set Metrics (Out-of-Sample):")
print(f"Test MSE: {mse_test:.7f}")
print(f"Test RMSE: {rmse_test:.7f}")
print(f"Test MAE: {mae_test:.7f}")

# Calcular el VIF de las variables de la regresión 1 con el índice optimizado
vif_data_optimized = calculate_vif(X_new.drop(columns=['const']))  # Excluir la constante para VIF
print("\nVariance Inflation Factor (VIF) después de optimizar INDEX2:")
print(vif_data_optimized)
vif_data_optimized.to_excel('vif_data_optimized.xlsx', index=False)

# Ensure y and INDEX2 are properly aligned and extracted from data
y = data['PRICE_Growth']  # Dependent variable
X = data[['INDEX2']]      # Independent variable as a DataFrame
X = sm.add_constant(X)    # Add a constant term for the intercept

# Fit the OLS model
model = sm.OLS(y, X).fit(cov_type='HC3')

# Calculate MSE
mse_index = mean_squared_error(y, model.predict(X))

# Print results
print("\nResultados del modelo OLS con INDEX2 como variable independiente (PRICE_Growth):")
print("--------------------------------------------------")
print(model.summary())
print(f"Mean Squared Error (MSE) between INDEX2 and PRICE_Growth: {mse_index:.4f}")
data['residuals'] = model.resid  # Guardar los residuos en el DataFrame

# 1. Basic Residual Statistics
residuals_mean = data['residuals'].mean()
residuals_std = data['residuals'].std()
residuals_skew = skew(data['residuals'].dropna())
residuals_kurtosis = kurtosis(data['residuals'].dropna())
dw_stat = durbin_watson(model.resid)

print(f"\nBasic Residual Statistics:")
print(f"Mean of Residuals: {residuals_mean:.7f} (Ideal: close to 0)")
print(f"Standard Deviation of Residuals: {residuals_std:.7f}")
print(f"Skewness of Residuals: {residuals_skew:.4f} (Ideal: close to 0)")
print(f"Kurtosis of Residuals: {residuals_kurtosis:.4f} (Ideal: close to 3 for normal distribution)")
print(f"Durbin-Watson Statistic: {dw_stat:.4f} (Ideal: close to 2, indicates no autocorrelation)")

# 2. Advanced Residual Diagnostics
# Residual Standard Error (RSE)
rse = np.sqrt(np.sum(model.resid ** 2) / (model.df_resid))

# Normality Tests
shapiro_stat, shapiro_p = shapiro(model.resid)
anderson_result = anderson(model.resid)
anderson_stat = anderson_result.statistic
anderson_critical_value_5 = anderson_result.critical_values[2]  # 5% significance level

print(f"\nAdvanced Residual Diagnostics:")
print(f"Residual Standard Error (RSE): {rse:.7f} (Measures typical prediction error)")
print(f"Shapiro-Wilk Test (p-value): {shapiro_p:.7f} (p > 0.05 indicates normality)")
print(f"Anderson-Darling Statistic: {anderson_stat:.7f} (Normality if < {anderson_critical_value_5} at 5%)")

# 4. Residual Influence and Outliers
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
leverage = influence.hat_matrix_diag

print(f"\nResidual Influence and Outliers:")
print(f"Max Cook's Distance: {np.max(cooks_d):.7f} (Values > 1 suggest influential points)")
print(f"Max Leverage Value: {np.max(leverage):.7f} (High values indicate potential outliers)")

# 5. Additional Error Metrics for Second Regression
y_pred = model.predict(X)
rmse_index = np.sqrt(mse_index)
mae_index = mean_absolute_error(y, y_pred)

print(f"\nAdditional Error Metrics for Second Regression:")
print(f"Root Mean Squared Error (RMSE): {rmse_index:.7f}")
print(f"Mean Absolute Error (MAE): {mae_index:.7f}")

# 6. Export Residuals and Diagnostics
residuals_df = pd.DataFrame({
    'Residuals': model.resid,
    'Fitted_Values': model.fittedvalues,
    'Cook_Distance': cooks_d,
    'Leverage': leverage
})
residuals_df.to_excel('residuals_diagnostics.xlsx', index=False)

print(f"\nResiduals and diagnostics exported to 'residuals_diagnostics.xlsx'")
print(f"Residual summary exported to 'residuals_summary.xlsx'")
print(max('INDEX2'))

# Exportar los mejores pesos a un archivo Excel
pd.DataFrame.from_dict(best_weights, orient='index', columns=['Weight']).to_excel('optimized_weights_bayesian.xlsx')

# Exportar el DataFrame con el índice optimizado
data.to_excel('data_FINAL.xlsx', index=False)

# Asegurarse de que TICKER y DATE estén como columnas antes de exportar
data.reset_index(inplace=True)

# Opcional: Guardar los residuos con TICKER y DATE en un archivo Excel
data[['TICKER', 'DATE', 'residuals']].to_excel('residuals_by_ticker_date.xlsx', index=False)

# --- Estandarizar PRICE_Growth por DATE usando MinMaxScaler ---
scaler_pg = MinMaxScaler()
data['PRICE_Growth_std'] = data.groupby('DATE', observed=True)['PRICE_Growth'].transform(
    lambda x: scaler_pg.fit_transform(x.values.reshape(-1, 1)).flatten()
)

# Eliminar filas con NaN en INDEX2 o PRICE_Growth_std
reg_std = data.dropna(subset=['INDEX2', 'PRICE_Growth_std'])

# Regresión entre INDEX2 y PRICE_Growth_std
X_std = sm.add_constant(reg_std['INDEX2'])
y_std = reg_std['PRICE_Growth_std']
model_std = sm.OLS(y_std, X_std).fit(cov_type='HC3')

# Métricas de la regresión estandarizada
mse_std = mean_squared_error(y_std, model_std.predict(X_std))
rmse_std = np.sqrt(mse_std)
mae_std = mean_absolute_error(y_std, model_std.predict(X_std))
r2_std = model_std.rsquared
adj_r2_std = model_std.rsquared_adj

print("\n--- Regresión INDEX2 vs PRICE_Growth estandarizado (MinMax por DATE) ---")
print(model_std.summary())
print(f"MSE: {mse_std:.6f}")
print(f"RMSE: {rmse_std:.6f}")
print(f"MAE: {mae_std:.6f}")
print(f"R²: {r2_std:.6f}")
print(f"R² ajustado: {adj_r2_std:.6f}")

# Gráfico de la regresión estandarizada
plt.figure(figsize=(8, 6))
plt.scatter(reg_std['INDEX2'], reg_std['PRICE_Growth_std'], color='purple', alpha=0.5, label='Observado')
x_vals_std = np.linspace(reg_std['INDEX2'].min(), reg_std['INDEX2'].max(), 100)
y_vals_std = model_std.params['const'] + model_std.params['INDEX2'] * x_vals_std
plt.plot(x_vals_std, y_vals_std, color='red', label='Línea de regresión')
plt.xlabel('INDEX2')
plt.ylabel('PRICE_Growth (MinMax por DATE)')
plt.title('Regresión: INDEX2 vs PRICE_Growth estandarizado')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('regresion_INDEX2_vs_PRICE_Growth_estandarizado.png')
plt.show()