from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# Cargar el conjunto de datos Wine
data = load_wine()

X = data.data 
y = data.target  


X_train_val, X_final, y_train_val, y_final = train_test_split(X, y, test_size=0.05, random_state=42)


X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42)

# Crear el modelo Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar y validar el modelo y los mismos pasos que con iris
rf.fit(X_train, y_train)


y_val_pred = rf.predict(X_val)


accuracy_val = accuracy_score(y_val, y_val_pred)
print(f"Accuracy en el conjunto de validación: {accuracy_val * 100:.2f}%")


y_final_pred = rf.predict(X_final)


accuracy_final = accuracy_score(y_final, y_final_pred)
print(f"Accuracy en el conjunto de evaluación final: {accuracy_final * 100:.2f}%")


df_final = pd.DataFrame(X_final, columns=data.feature_names)  
df_final['class'] = y_final  



excel_filename = 'evaluacion_wine.xlsx'
df_final.to_excel(excel_filename, index=False)


joblib.dump(rf, 'modelo_rf_wine.pkl')

print(f"El archivo con los resultados de la evaluación final se ha guardado como '{excel_filename}'")
