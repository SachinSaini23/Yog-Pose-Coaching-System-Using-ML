import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model

# Initialize necessary variables
is_init = False
X, y = None, None
label = []
dictionary = {}
c = 0

# Iterate over the files and process the .npy files
for i in os.listdir():
    if i.endswith(".npy") and not i.startswith("labels"):  # More precise file filtering
        if not is_init:
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]]*size).reshape(-1,1)
            is_init = True
        else:
            new_X = np.load(i)
            X = np.concatenate((X, new_X), axis=0)
            size = new_X.shape[0]
            new_y = np.array([i.split('.')[0]]*size).reshape(-1,1)
            y = np.concatenate((y, new_y), axis=0)

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c += 1

# Replace string labels with numerical labels using the dictionary
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# One-hot encode the labels
y = to_categorical(y)

# Shuffle the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X_new = X[indices]
y_new = y[indices]

# Build the neural network model
input_layer = Input(shape=(X.shape[1],))  # Adjust input shape based on your data
m = Dense(128, activation="tanh")(input_layer)
m = Dense(64, activation="tanh")(m)
output_layer = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

# Train the model
model.fit(X_new, y_new, epochs=80)

# Save the model and label dictionary
model.save("model.h5")
np.save("labels.npy", np.array(label, dtype=object))  # Saving as object to handle string labels
