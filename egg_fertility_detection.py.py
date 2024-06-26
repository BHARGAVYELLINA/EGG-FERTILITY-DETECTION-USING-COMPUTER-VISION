import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

images_dir = 'images'

def check_directory(path,):
    if not os.path.exists(path):
        raise ValueError(f"{path} does not exist.")
    if not os.listdir(path):
        raise ValueError(f"{path} is empty.")
    print(f"{path} is valid and contains files.")

check_directory(images_dir)

def load_and_preprocess_data(directory):
    images = []
    labels = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            img = load_img(os.path.join(directory, filename), target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            images.append(img_array)
            filename_label = filename.split('_')[0].strip()
            labels.append(1 if filename_label.lower() == "fertile" else 0)
            filenames.append(filename)
    return np.array(images), np.array(labels), filenames

model_file = 'egg_fertility_model.keras'
if os.path.exists(model_file):
    model = load_model(model_file)
    print("Model loaded successfully.")
else:
    print(f"Model file '{model_file}' not found. Training the model.")

    images, labels, filenames = load_and_preprocess_data(images_dir)
    
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    log_dir = "logs/fit/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(images, labels, epochs=10, callbacks=[tensorboard_callback])

    model.save('egg_fertility_model.keras')
    print("Model trained and saved successfully.")

    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.show()

if 'model' in locals():
    images, labels, filenames = load_and_preprocess_data(images_dir)
    
    predictions = model.predict(images)
    predicted_labels = (predictions > 0.5).astype(int).flatten()
    
    for filename, prediction in zip(filenames, predictions):
        fertility = "Fertile" if prediction > 0.5 else "Infertile"
        print(f"Image: {filename}, Prediction: {fertility} ({prediction[0]:.4f})")
    
    cm = confusion_matrix(labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Infertile', 'Fertile'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
else:
    print("Model not available for inference.")
