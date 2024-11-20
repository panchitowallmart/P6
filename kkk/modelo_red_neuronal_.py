from preparar_datos import preparar_datos
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Cargar datos preparados
X_train, X_test, y_train, y_test, X_val, y_val = preparar_datos()

# Vectorización del texto
vectorizer = TfidfVectorizer(max_features=5000)  # Limitar a 5000 características
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()
X_val_vec = vectorizer.transform(X_val).toarray()

# Crear la red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5000,)),  # Tamaño del vector TF-IDF
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),  # Dropout para prevenir sobreentrenamiento
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 etiquetas: Positive, Negative, Neutral, Irrelevant
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks para prevenir sobreentrenamiento
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitorea la pérdida en validación
    patience=3,          # Detén si no mejora en 3 épocas consecutivas
    restore_best_weights=True  # Restaura los mejores pesos al final
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',  # Monitorea la pérdida en validación
    factor=0.5,          # Reduce la tasa de aprendizaje a la mitad
    patience=2           # Si no mejora en 2 épocas consecutivas
)

# Entrenar el modelo
model.fit(
    X_train_vec, y_train,
    validation_data=(X_test_vec, y_test),
    epochs=50,  # Máximo número de épocas
    batch_size=32,
    callbacks=[early_stopping, reduce_lr]  # Agregar los callbacks
)
# Guardar el modelo entrenado
model.save("modelo ya entrenado.h5")
print("Modelo guardado como 'modelo ya entrenado.h5'")


# Evaluar en el conjunto de validación
predictions = model.predict(X_val_vec)
predicted_labels = predictions.argmax(axis=1)

# Reporte de clasificación
print(classification_report(y_val, predicted_labels))

# Generar matriz de confusión
cm = confusion_matrix(y_val, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive", "Negative", "Neutral", "Irrelevant"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.show()

# Generar gráficos ROC
n_classes = 4  # Cambia este número si tienes más o menos clases
fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    y_true = (np.array(y_val) == i).astype(int)  # Convertir a binario para cada clase
    fpr[i], tpr[i], _ = roc_curve(y_true, predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Graficar ROC
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"Clase {i} (Área = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--")  # Línea base
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Gráfico ROC por Clase")
plt.legend(loc="lower right")
plt.show()
