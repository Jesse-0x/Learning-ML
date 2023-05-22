# Preprocess the data by removing noise and splitting it into training and testing sets.
# Build a Convolutional Neural Network (CNN) to extract features from the input data.
# Build a Recurrent Neural Network (RNN) to capture the temporal dependencies in the data.
# Build an Encoder-Decoder architecture to map the input data to a lower-dimensional latent space and then decode it back to the original space.
# Use Latent Dirichlet Allocation (LDA) to model the noise in the data and remove it from the input.
# Train the model on the training set and evaluate it on the testing set.
# Use the model to reconstruct the test set and visualize the reconstructed data.

# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, LSTM, RepeatVector, Dense, Lambda
from tensorflow.keras.models import Model
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import pickle

# Preprocess the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Build the CNN
input_img = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the CNN
print(f"Training the CNN... x_train_noisy.shape = {x_train_noisy.shape}, x_train.shape = {x_train.shape}")
try:
    autoencoder.load_weights('autoencoder_cnn.h5')
except:
    # increase the learning rate
    autoencoder.fit(x_train_noisy, x_train,
                    epochs=2,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test))

# Save the CNN
autoencoder.save('autoencoder_cnn.h5')





# Build the RNN
timesteps = 28
input_dim = 28
latent_dim = 2
inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)
sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)
# learning rate too large
sequence_autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy')

# Train the RNN
try:
    sequence_autoencoder.load_weights('autoencoder_rnn.h5')
except:
    print(f"Training the RNN... x_train_noisy.shape = {x_train_noisy.shape}, x_train.shape = {x_train.shape}")
    sequence_autoencoder.fit(x_train_noisy, x_train,
                             epochs=50,
                             batch_size=128,
                             shuffle=True,
                             validation_data=(x_test_noisy, x_test))

# Save the RNN
sequence_autoencoder.save('autoencoder_rnn.h5')






# Build the Encoder-Decoder
inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)
sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)
sequence_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the Encoder-Decoder
print(f"Training the Encoder-Decoder... x_train_noisy.shape = {x_train_noisy.shape}, x_train.shape = {x_train.shape}")
try:
    sequence_autoencoder.load_weights('autoencoder_encoder_decoder.h5')
except:
    sequence_autoencoder.fit(x_train_noisy, x_train,
                             epochs=50,
                             batch_size=128,
                             shuffle=True,
                             validation_data=(x_test_noisy, x_test))

# Save the Encoder-Decoder
sequence_autoencoder.save('autoencoder_encoder_decoder.h5')

# Show the results
decoded_imgs = sequence_autoencoder.predict(x_test_noisy)
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display noisy
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display denoised
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()






# Build the LDA
lda = LatentDirichletAllocation(n_components=2, random_state=0 )

# Train the LDA
x_train_noisy_lda = x_train_noisy.reshape((x_train_noisy.shape[0], x_train_noisy.shape[1] * x_train_noisy.shape[2]))
print(f"Training the LDA... x_train_noisy.shape = {x_train_noisy.shape}, x_train_noisy_lda.shape = {x_train_noisy_lda.shape}")
try:
    lda = pickle.load(open('lda.pkl', 'rb'))
except:
    # do verbose
    lda.fit(x_train_noisy_lda)

# Save the LDA
pickle.dump(lda, open('lda.pkl', 'wb'))




# Build the Denoising Autoencoder
inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)
sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)
sequence_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the Denoising Autoencoder
print(
    f"Training the Denoising Autoencoder... x_train_noisy.shape = {x_train_noisy.shape}, x_train.shape = {x_train.shape}")
try:
    sequence_autoencoder.load_weights('sequence_autoencoder.h5')
except:
    sequence_autoencoder.fit(x_train_noisy, x_train,
                             epochs=50,
                             batch_size=128,
                             shuffle=True,
                             validation_data=(x_test_noisy, x_test))

# Save the Denoising Autoencoder
sequence_autoencoder.save('sequence_autoencoder.h5')

# Evaluate the Denoising Autoencoder
sequence_autoencoder.evaluate(x_test_noisy, x_test)

# Reconstruct the test set
x_test_encoded = encoder.predict(x_test_noisy)
x_test_reconstructed = sequence_autoencoder.predict(x_test_noisy)

# Visualize the reconstructed data
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Display noisy
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test_reconstructed[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test_reconstructed[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

