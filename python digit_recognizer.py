import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# 1. Load the Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 2. Build the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(10, activation='softmax')
])

# 3. Compile and Train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 🚨 EDIT THIS LINE: Change the number of epochs to commit 
# a new version to GitHub! (e.g., 3, 5, 8, 10, etc.)
epochs_to_run = 122

print(f"Starting training for {epochs_to_run} epochs...")
model.fit(x_train, y_train, epochs=epochs_to_run)

# 4. Evaluate (Optional)
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nModel trained for {epochs_to_run} epochs. Test accuracy: {acc*100:.2f}%")

# 5. Save the model (For portfolio evidence)
model.save(f'mnist_model_e{epochs_to_run}.h5')
print(f"Model saved as mnist_model_e{epochs_to_run}.h5")