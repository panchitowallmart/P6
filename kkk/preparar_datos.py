import pandas as pd
import re
from sklearn.model_selection import train_test_split

def preparar_datos():
    # Rutas de los archivos proporcionadas
    training_file_path = r"C:\Users\emili\Downloads\archive\twitter_training.csv"
    validation_file_path = r"C:\Users\emili\Downloads\archive\twitter_validation.csv"

    # Cargar los datos desde las rutas proporcionadas
    training_data = pd.read_csv(training_file_path)
    validation_data = pd.read_csv(validation_file_path)

    # Renombrar columnas relevantes
    training_data = training_data.rename(columns={training_data.columns[-2]: "label", training_data.columns[-1]: "text"})
    validation_data = validation_data.rename(columns={validation_data.columns[-2]: "label", validation_data.columns[-1]: "text"})

    # Función para limpiar el texto
    def clean_text(text):
        if not isinstance(text, str):
            return ""  # Devuelve un texto vacío si no es una cadena
        text = re.sub(r"http\S+", "", text)  # Quitar URLs
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Quitar caracteres especiales
        text = text.lower().strip()  # Convertir a minúsculas y quitar espacios
        return text

    # Limpiar los textos asegurando que no haya valores nulos o no textuales
    training_data['text'] = training_data['text'].fillna("").apply(clean_text)
    validation_data['text'] = validation_data['text'].fillna("").apply(clean_text)

    # Convertir etiquetas a valores numéricos
    label_mapping = {"Positive": 1, "Negative": 0, "Neutral": 2, "Irrelevant": 3}
    training_data['label'] = training_data['label'].map(label_mapping)
    validation_data['label'] = validation_data['label'].map(label_mapping)

    # Dividir los datos de entrenamiento para validación interna si es necesario
    X_train, X_test, y_train, y_test = train_test_split(
        training_data['text'], training_data['label'], test_size=0.2, random_state=42
    )

    # Validación
    X_val, y_val = validation_data['text'], validation_data['label']

    return X_train, X_test, y_train, y_test, X_val, y_val
