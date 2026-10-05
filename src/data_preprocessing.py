import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from settings import DATA_DIR, IMG_SIZE, BATCH_SIZE

def load_data(data_dir, img_size, test_size=0.2, val_size=0.25, batch_size=BATCH_SIZE, augment=False):
    X = []
    y = []
    labels = os.listdir(data_dir)
    label_to_index = {label: i for i, label in enumerate(labels)}

    for label in labels:
        label_path = os.path.join(data_dir, label)
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = load_img(image_path, target_size=(img_size, img_size))
            image = img_to_array(image) / 255.0 
            X.append(image)
            y.append(label_to_index[label])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)

    if augment:
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        train_datagen.fit(X_train)
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        return train_generator, X_val, y_val, X_test, y_test

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    data_dir = DATA_DIR + "/data"
    img_size = IMG_SIZE
    train_generator, X_val, y_val, X_test, y_test = load_data(data_dir, img_size, augment=True)
