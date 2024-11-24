import joblib
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


def index(request):
    if request.method == 'POST':

        # Otra verificación de que se cargaron ambos archivos por si acaso
        if 'excel_file' not in request.FILES or 'model_file' not in request.FILES:
            return HttpResponse("Por favor sube tanto un archivo Excel como un archivo de modelo preentrenado (.pkl)")

        excel_file = request.FILES['excel_file']
        model_file = request.FILES['model_file']


        # Intentar cargar el modelo desde el archivo .pkl
        try:
            model = joblib.load(model_file)
            # Verificar si el modelo cargado es valido
            if not isinstance(model, BaseEstimator):
                return HttpResponse("El archivo subido no contiene un modelo válido de scikit-learn.")
        except Exception as e:
            return HttpResponse(f"Error al cargar el modelo: {str(e)}")
        
        # Intentar leer el archivo de Excel
        try:
            df = pd.read_excel(excel_file)
        except Exception as e:
            return HttpResponse(f"Error al leer el archivo Excel: {str(e)}")

        # Verificar que el archivo tiene al menos una fila de datos para predicción
        if df.empty:
            return HttpResponse("El archivo Excel no contiene datos.")

        # Asumimos que el archivo de excel tiene todas las caracteristicas del modelo y al final la clase correcta para poder comparar con laa predicción
        features = df.columns[:-1]  
        X = df[features]  
        y = df.iloc[:, -1]  

        try:
            # Hacer la predicción con el modelo cargado
            predictions = model.predict(X)

            # Calcular la precisión (porcentaje de acierto)
            accuracy = accuracy_score(y, predictions) * 100  # Multiplicamos por 100 para obtener el porcentaje

            # Agregar las predicciones al DataFrame
            df['Predicción'] = predictions

            # Mostrar resultados en la tabla HTML y pasar la precisión
            return render(request, 'predict/resultados.html', {
                'data': df.to_html(index=False),
                'accuracy': f'{accuracy:.2f}%'
            })
        except Exception as e:
            return HttpResponse(f"Error al hacer la predicción: {str(e)}")

    return render(request, 'predict/index.html')
