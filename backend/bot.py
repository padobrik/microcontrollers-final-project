# Импорт необходимых библиотек
from telegram.ext import *
from io import BytesIO
import cv2
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam

# Функция для инициализации старта бота
def start(update, context):
    update.message.reply_text('Привет! Давай лечиться!')

# Функция для инициализации помощи по командам бота
def help(update, context):
    update.message.reply_text('''
    /start - начать общение
    /help - показать это сообщение
    Для того, чтобы провести экспресс-диагностику, просто
    прикрепи изображение :)
    ''')

# Обработчик получаемых изображений
def handle_photo(update, context):
    # получение изображения на сервер и его преобразование в массив
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)

    # декодинг, преобразование цвета и размера изображения
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    # импорт структуры нейронной сети и структуры весов для каждого нейрона
    json_file = open('backend/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("backend/model.h5")
    loaded_model.compile(optimizer = Adam(lr) ,
              loss = "binary_crossentropy", 
              metrics=["accuracy"])

    # определение категорий изображений
    classes = {
        0: 'доброкачественная опухоль',
        1: 'меланома'
    }

    # предсказание и вероятности
    prediction = loaded_model.predict(np.array([img / 255]))
    probability = float(prediction[0][np.argmax(prediction)]) * 100

    # возврат сообщения с ответом
    update.message.reply_text(f'Мне кажется, что это {classes[np.argmax(prediction)]} с вероятностью {round(probability, 2)}%')

# Открываем файл с токеном для запуска бота
with open('client/token.txt', 'r') as f:
    TOKEN = f.read()

# Фиксируем размер градиентного шага для нейронной сети
lr = 1e-5

# Создаем объект класса Updater на основании токена == репрезентация бота
# Также извлекаем из него диспетчера - буквально получателя сообщений от пользователя
updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher

# Добавляем командные объекты и объекты сообщений
# Первые определяют поведение бота на основании команд, подаваемыми пользователем
# Вторые определяют получение изображение и возврат ответа
dp.add_handler(CommandHandler('start', start))
dp.add_handler(CommandHandler('help', help))
dp.add_handler(MessageHandler(Filters.photo, handle_photo))

# Запуск бота и внутренней среды обработки
updater.start_polling()
updater.idle()