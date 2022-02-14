import tensorflow as tf
from data import get_train, get_test
import matplotlib.pyplot as plt

train_X, train_y = get_train()
test_X, test_y = get_test()

cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(64,64,1), padding='same'), # 64, 64
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2), # 32, 32, 32
    tf.keras.layers.Flatten(), # 1D vector of 32*32*32
    tf.keras.layers.Dense(units=16, activation='relu'), 
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

cnn.summary()
cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['binary_accuracy']
)

history = cnn.fit(train_X, train_y, batch_size=32, verbose=1, epochs=20, validation_data=(test_X,test_y), validation_batch_size=32)

cnn.save('cls')

plt.plot(list(range(len(history.history['loss']))), history.history['loss'], 'g-o', label='Training')
plt.plot(list(range(len(history.history['val_loss']))), history.history['val_loss'], 'r-o', label='Validation')
plt.savefig('losses.png')
plt.show()

plt.plot(list(range(len(history.history['binary_accuracy']))), history.history['binary_accuracy'], 'g-o', label='Training')
plt.plot(list(range(len(history.history['val_binary_accuracy']))), history.history['val_binary_accuracy'], 'r-o', label='Validation')
plt.savefig('binary_accuracies.png')
plt.show()