import cv2
import numpy as np
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Parameters
working_dir = 'shapes'
img_size = 200  # Size of image fed into model

def flatten(dimData, images):
    images = np.array(images)
    images = images.reshape(len(images), dimData)
    images = images.astype('float32')
    images /= 255
    return images

# Load data
folders = ['triangle', 'star', 'square', 'circle', 'rectangle', 'ellipse', 'rounded_rectangle']
labels = []
images = []

for folder in folders:
    print(f"Loading {folder} images...")
    for path in os.listdir(os.path.join(working_dir, folder)):
        img = cv2.imread(os.path.join(working_dir, folder, path), 0)
        images.append(cv2.resize(img, (img_size, img_size)))
        labels.append(folders.index(folder))

# Break data into training and test sets
train_images, test_images = [], []
train_labels, test_labels = [], []
to_train = 0

for image, label in zip(images, labels):
    if to_train < 5:
        train_images.append(image)
        train_labels.append(label)
        to_train += 1
    else:
        test_images.append(image)
        test_labels.append(label)
        to_train = 0

# Flatten data
dataDim = np.prod(train_images[0].shape)
train_data = flatten(dataDim, train_images)
test_data = flatten(dataDim, test_images)

# Convert labels to categorical
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# Define the model
model = Sequential()
model.add(Dense(256, activation='tanh', input_shape=(dataDim,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(folders), activation='softmax'))

# Compile and train the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=50, verbose=1, validation_data=(test_data, test_labels_one_hot))

# Evaluate the model
[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print(f"Evaluation result on Test Data : Loss = {test_loss}, accuracy = {test_acc}")

# Save the model
model.save('shapes_model.h5')
