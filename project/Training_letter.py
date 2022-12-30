import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from load_data import load_data, data_shuffle

X_train, Y_train, X_test, Y_test = load_data()

X_train, Y_train = data_shuffle(X_train, Y_train)
X_train = X_train.astype('float32') / 255
X_train = np.expand_dims(X_train, -1)
Y_train = to_categorical(Y_train)

X_test, Y_test = data_shuffle(X_test, Y_test)
X_test = X_test.astype('float32') / 255
X_test = np.expand_dims(X_test, -1)
Y_test = to_categorical(Y_test)

callback = EarlyStopping(monitor="val_loss", patience=15)

network = Sequential([
    Conv2D(32, (3, 3), input_shape=(100, 100, 1), padding='same', activation='relu'),
    Conv2D(32, (5, 5), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Conv2D(64, (5, 5), activation='relu', padding='same'),
    Conv2D(128, (5, 5), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(3, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Dropout(0.35),
    MaxPooling2D(pool_size=(3, 3)),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(Y_train.shape[1], activation='softmax')
])
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(network.summary())

history = network.fit(X_train, Y_train, epochs=30, validation_split=0.2, batch_size=300)

print("\nTesting ...")
loss, accuracy = network.evaluate(X_train, Y_train, verbose=0)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
loss, accuracy = network.evaluate(X_test, Y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))

network.save("all.h5")

# 顯示訓練和驗證損失
loss = history.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "bo-", label="Training Loss")
plt.plot(epochs, val_loss, "ro--", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# 顯示訓練和驗證準確度
acc = history.history["accuracy"]
epochs = range(1, len(acc)+1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo-", label="Training Acc")
plt.plot(epochs, val_acc, "ro--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
print()
