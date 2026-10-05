import matplotlib.pyplot as plt
from data_preprocessing import load_data
from model import build_model
from settings import DATA_DIR, EPOCHS, BATCH_SIZE, IMG_SIZE

def train_model(data_dir, img_size, augment=True, epochs=10, batch_size=32):
    # Загрузка данных
    if augment:
        train_generator, X_val, y_val, X_test, y_test = load_data(data_dir, img_size, augment=True)
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_dir, img_size, augment=False)

    # Создание модели
    input_shape = (img_size, img_size, 3)
    num_classes = 5
    model = build_model(input_shape, num_classes)



    # Обучение модели
    if augment:
        history = model.fit(train_generator,
                            epochs=epochs,
                            validation_data=(X_val, y_val))
    else:
        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val))

    # Оценка производительности модели
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    print("Точность проверки:", val_accuracy)
    print("Точность испытаний", test_accuracy)

    # Построение графиков
    plt.plot(history.history['accuracy'], label='Точность на обучающем этапе')
    plt.plot(history.history['val_accuracy'], label='Точность на валидационном этапе')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Потери на обучающем этапе')
    plt.plot(history.history['val_loss'], label='Потери на валидационном этапе')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.show()

    return model  # Возвращаем модель

if __name__ == "__main__":
    data_dir = DATA_DIR + "/data"
    img_size = IMG_SIZE
    trained_model = train_model(data_dir, img_size, augment=True, epochs=EPOCHS, batch_size=BATCH_SIZE)
    trained_model.save_weights(DATA_DIR + "/FLOWERS.weights.h5")
