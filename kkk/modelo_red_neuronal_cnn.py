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

# Ajustar los datos para ser compatibles con CNN
X_train_vec = X_train_vec.reshape(-1, 5000, 1)
X_test_vec = X_test_vec.reshape(-1, 5000, 1)
X_val_vec = X_val_vec.reshape(-1, 5000, 1)

# Crear el modelo CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(5000, 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 etiquetas: Positive, Negative, Neutral, Irrelevant
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks para ajustar el learning rate (sin EarlyStopping)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2
)

# Entrenar el modelo
model.fit(
    X_train_vec, y_train,
    validation_data=(X_test_vec, y_test),
    epochs=5,  # Entrenar siempre 50 épocas
    batch_size=32,
    callbacks=[reduce_lr]  # Solo ajustar el learning rate, sin detener el entrenamiento
)

# Guardar el modelo entrenado
model.save("modelo_cnn_50_epocas.h5")
print("Modelo guardado como 'modelo_cnn_50_epocas.h5'")

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
