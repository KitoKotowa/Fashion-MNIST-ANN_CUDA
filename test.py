import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import fashion_mnist

# Define constants
INPUT_SIZE = 784
HIDDEN1_SIZE = 128
HIDDEN2_SIZE = 128
OUTPUT_SIZE = 10
LEARNING_RATE = 0.01
EPOCHS = 20
BATCH_SIZE = 128

# Load Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the data

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into 784
    Dense(HIDDEN1_SIZE, activation='relu'),  # First hidden layer
    Dense(HIDDEN2_SIZE, activation='relu'),  # Second hidden layer
    Dense(OUTPUT_SIZE, activation='softmax')  # Output layer
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_accuracy:.2f}")

