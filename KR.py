import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Укажите путь к папке с данными
data_dir = 'D:/ЯРИК/УЧЕБА/ММО/Dataset_flowers'

# Получаем список всех классов (подпапок)
classes = [class_name for class_name in os.listdir(data_dir) if not class_name.startswith('.')]

images = []

labels = []


for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            images.append(image_path)
            labels.append(class_name)

# Преобразуем метки в числовой формат
label_to_index = {class_name: i for i, class_name in enumerate(classes)}
labels = [label_to_index[label] for label in labels]

# Разбиваем данные на обучающую, валидационную и тестовую выборки
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Размер изображений, к которому мы хотим привести исходные изображения
image_size = (150, 150)

# Создаем генераторы изображений для обучения, валидации и тестирования
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    os.path.join(data_dir, 'validation'),
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)