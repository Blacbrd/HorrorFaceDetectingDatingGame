import keras
import numpy as np
from sklearn.utils import class_weight


# Add numpy arrays to file
x_train = np.load("C:\\Users\\blacb\\Desktop\\UniversityGitFolder\\tkinterpythongame\\NumpyArrays\\EmotionTrainingData.npy")
y_train = np.load("C:\\Users\\blacb\\Desktop\\UniversityGitFolder\\tkinterpythongame\\NumpyArrays\\EmotionLabels.npy")

# Normalising data allows us to put our data on a common scale
# This also prevents very large or very small values from appearing
# This will make all the values go between 0 - 1
x_train = keras.utils.normalize(x_train, axis=1)

# Remove class imbalance between having more or less data
classWeights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
classWeights = dict(enumerate(classWeights))

# Feed forward neural network
model = keras.models.Sequential()

# This is the input layer of the neural network, it takes in the whole image
# Input shape is width x height x colour depth, 100x100x1
# We don't need to include the -1, so we use [1:] to remove it
model.add(keras.layers.Input(shape=x_train.shape[1:]))

# 64 is the amount of filters/kernels we use, each finding different types of patterns
# (3, 3) is the size of the kernel matrix. Can also just use 3
# Strides is how much we want to slide over the image each time, 1,1 just means over by one pixel
# Padding is the values that the kernel will take when it overshoots the image (goes into the void where there is no image)
# Non linearity introduced through relu
# Non linearity will allow our model and its categorisation to be more flexible
model.add(keras.layers.Conv2D(64, (3, 3), strides=(1,1), padding="same"))


model.add(keras.layers.Activation("relu"))

# Pooling reduces the size/resolution of our image while still maintaining important information
# (2,2) essentially divides both of our dimensions by 2, 50 x 50 (x1 for colour)
model.add(keras.layers.MaxPool2D((2,2)))

# Drop 10% of neurones after pooling
# This reduces the chance of a single feature being too relied upon
model.add(keras.layers.Dropout(0.1))

# Now we add another one of the same layer
model.add(keras.layers.Conv2D(128,(5,5), padding='same'))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.1))


model.add(keras.layers.Conv2D(256, (3, 3), padding="same"))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D((2,2)))
model.add(keras.layers.Dropout(0.1))

# Make the image into a 1D array, allows us to use pixels as neurone input layer
model.add(keras.layers.Flatten())

# Fully connected layer, allows all neurons to be considered during output
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dropout(0.2))

# We have 2 outputs between 0 - 1, so we have 2 output neurons
# We use softmax as it returns every value to be between 0 - 1, and also has them all add to 1
# This makes it so that they can be interpretted as probabilities
# Softmax also pushes higher values further from the others, encouraging the NN to increase its confidence
model.add(keras.layers.Dense(2, activation="softmax"))

# Adam is a very popular optimisation model, especially for image identification
# Since I am using many categories that are categorised using numbers (0, 1, 2, 3), SCC is used to prevent converting labels to one-hot encoded format
# One hot encoding is usually used for string labels (so if I used the words "happy", "sad" as labels, which would be converted to matrices)
# Metrics being accuracy so we can see how well our model is performing
# Lower learning rate to avoid being in local minima
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# This stops the model when the validation loss stops decreasing for 2 epochs
# It then restores the time for when the model was at its best
# This prevents overfitting data
earlyStopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

# Training:
# Epochs is how many times the model is trained with previous weights
# Batch size increases training speed, and introduces noise which can help escape local minima in gradient descent
# Validation split is the % of training and validation data, used to reduce loss faster
# Added class weights to help the data inbalance
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[earlyStopping], class_weight=classWeights)

# Save the mode
model.save("EmotionRecogniser.h5")
