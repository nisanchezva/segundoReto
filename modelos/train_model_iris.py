from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

data = load_iris()


X = data.data  # Características
y = data.target  # Etiquetas



print("Características:", data.feature_names)


print("Clases:", data.target_names)

#Usaremos el 95% de los datos para el entrenamiento, validación y el 5 % para la prueba final dentro de la aplicación
X_train_val, X_final, y_train_val, y_final = train_test_split(X, y, test_size=0.05, random_state=42)


X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42)

# Usaremos Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

#Entrenando y evlauando el modelo
rf.fit(X_train, y_train)


y_val_pred = rf.predict(X_val)


accuracy_val = accuracy_score(y_val, y_val_pred)
print(f"Accuracy en el conjunto de validación: {accuracy_val * 100:.2f}%")

# Hacer predicciones en el conjunto de evaluación final (5% de los datos)
y_final_pred = rf.predict(X_final)

# Evaluar el rendimiento en el conjunto de evaluación final
accuracy_final = accuracy_score(y_final, y_final_pred)
print(f"Accuracy en el conjunto de evaluación final: {accuracy_final * 100:.2f}%")

# Guardamos el 5 % en un excel para usarlo en la aplicación
df_final = pd.DataFrame(X_final, columns=data.feature_names)  
df_final['class'] = y_final  


excel_filename = 'evaluacion_iris.xlsx'
df_final.to_excel(excel_filename, index=False)

#Guardamos el modelo
joblib.dump(rf, 'modelo_rf_iris.pkl')


print(f"El archivo con los resultados de la evaluación final se ha guardado como '{excel_filename}'")
