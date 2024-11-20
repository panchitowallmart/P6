import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from preparar_datos import preparar_datos
from tkinter import Tk, Label
from PIL import Image, ImageTk
import os

# Ruta completa del modelo guardado
modelo_guardado = "D:/kkk/modelo ya entrenado.h5"

# Verificar si el modelo existe
if not os.path.exists(modelo_guardado):
    print(f"Error: El modelo '{modelo_guardado}' no se encuentra.")
    exit()

# Cargar el modelo guardado
model = tf.keras.models.load_model(modelo_guardado)
print(f"Modelo cargado desde '{modelo_guardado}'")

# Ajustar el vectorizador con los datos de entrenamiento
X_train, _, _, _, _, _ = preparar_datos()
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(X_train)

# Pedir al usuario una frase para analizar
input_text = input("Escribe una frase para analizar: ")

# Vectorizar la frase
input_vec = vectorizer.transform([input_text]).toarray()

# Realizar la predicción
prediction = model.predict(input_vec)
print(f"Probabilidades predichas por el modelo: {prediction}")

# Establecer umbrales para Positive y Negative
positive_threshold = 0.5
negative_threshold = 0.5

# Revisar probabilidades predichas
if prediction[0][1] >= positive_threshold:  # Clase 1 es "Positive"
    predicted_label_corrected = "Positive"
elif prediction[0][0] >= negative_threshold:  # Clase 0 es "Negative"
    predicted_label_corrected = "Negative"
else:
    predicted_label_corrected = "Neutral"

# Mostrar la predicción
print(f"La frase fue clasificada como: {predicted_label_corrected}")

# Verificar si las imágenes existen
happy_path = "D:/kkk/happy.png"
sad_path = "D:/kkk/sad.png"

if not os.path.exists(happy_path):
    print(f"Error: 'happy.png' no encontrado en {happy_path}")
    exit()
if not os.path.exists(sad_path):
    print(f"Error: 'sad.png' no encontrado en {sad_path}")
    exit()

# Mostrar una imagen basada en la predicción
def show_image(image_path):
    # Crear una ventana gráfica
    root = Tk()
    root.title("Resultado de Sentimiento")
    
    # Cargar y mostrar la imagen
    img = Image.open(image_path)
    img = img.resize((200, 200), Image.LANCZOS)  # Cambiado de ANTIALIAS a LANCZOS
    img = ImageTk.PhotoImage(img)
    
    label = Label(root, image=img)
    label.image = img
    label.pack()
    
    # Ejecutar la ventana
    root.mainloop()

# Mostrar la imagen correspondiente
if predicted_label_corrected == "Positive":
    show_image(happy_path)  # Ruta completa para happy.png
elif predicted_label_corrected == "Negative":
    show_image(sad_path)  # Ruta completa para sad.png
else:
    print("No se mostrará ninguna imagen para esta clasificación.")
