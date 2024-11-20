from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import numpy as np

# Cargar el modelo guardado
modelo_guardado = "modelo_cnn_entrenado.h5"
model = tf.keras.models.load_model(modelo_guardado)
print(f"Modelo cargado desde '{modelo_guardado}'")

# Simular el vectorizador original (TF-IDF con 5000 características)
vectorizer = TfidfVectorizer(max_features=5000)
# Ajuste temporal para que funcione con las dimensiones correctas
dummy_texts = ["placeholder"] * 5000  # Textos ficticios para llenar las 5000 características
vectorizer.fit(dummy_texts)

# Textos nuevos para probar
new_texts = [
    "I love this product! It's amazing!",  # Ejemplo positivo
    "This is the worst experience ever.",  # Ejemplo negativo
    "I'm not sure how I feel about this.",  # Ejemplo neutral
    "Totally unrelated to the topic.",      # Ejemplo irrelevante
]

# Vectorizar los nuevos textos
new_texts_vec = vectorizer.transform(new_texts).toarray().reshape(-1, 5000, 1)

# Predecir las clases
predictions = model.predict(new_texts_vec)
predicted_labels = predictions.argmax(axis=1)

# Etiquetas correspondientes
labels = ["Positive", "Negative", "Neutral", "Irrelevant"]

# Mostrar resultados
print("\nResultados de predicción:")
for i, text in enumerate(new_texts):
    print(f"Texto: {text}")
    print(f"Predicción: {labels[predicted_labels[i]]}")
    print("-" * 50)
