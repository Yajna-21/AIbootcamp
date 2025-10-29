from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt

# Load and preprocess Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# -------------------------
# Model 1: Base
# -------------------------
model_base = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model_base.compile(optimizer=Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

history_base = model_base.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
loss, test_accuracy_base = model_base.evaluate(x_test, y_test)

# -------------------------
# Model 2: L2 = 1e-4
# -------------------------
model_le4 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
    Dense(10, activation='softmax')
])

model_le4.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

history_le4 = model_le4.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
loss, test_accuracy_le4 = model_le4.evaluate(x_test, y_test)

# -------------------------
# Model 3: L2 = 1e-2
# -------------------------
model_le2 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu', kernel_regularizer=l2(1e-2)),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=l2(1e-2)),
    Dense(10, activation='softmax')
])

model_le2.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

history_le2 = model_le2.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
loss, test_accuracy_le2 = model_le2.evaluate(x_test, y_test)

# -------------------------
# Visualization
# -------------------------
plt.figure(figsize=(8,5))
plt.plot(history_base.history['val_accuracy'], label='Base Model', color='blue')
plt.plot(history_le4.history['val_accuracy'], label='L2=1e-4', color='green')
plt.plot(history_le2.history['val_accuracy'], label='L2=1e-2', color='red')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Print test accuracies
print(f"Base Model Test Accuracy: {test_accuracy_base:.4f}")
print(f"L2=1e-4 Model Test Accuracy: {test_accuracy_le4:.4f}")
print(f"L2=1e-2 Model Test Accuracy: {test_accuracy_le2:.4f}")
