import numpy as np
import os
from tensorflow.keras.preprocessing import image
from model import build_model
from settings import IMG_SIZE, DATA_DIR

def load_trained_model(weights_path):
    input_shape = (IMG_SIZE, IMG_SIZE, 3)  # Размер изображений после предобработки
    num_classes = 5  # Количество классов
    model = build_model(input_shape, num_classes)
    model.load_weights(weights_path)
    return model


def classify_image(model, image_path, class_names):
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class]

    return predicted_class_name, predictions

if __name__ == "__main__":
    weights_path = DATA_DIR + "/FLOWERS.weights.h5"
    model = load_trained_model(weights_path)

    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    image_path = DATA_DIR + "/test.jpg"
    predicted_class, predictions = classify_image(model, image_path, class_names)
    print("Прогнозируемый класс:", predicted_class)
    print("Прогнозы:", predictions)
