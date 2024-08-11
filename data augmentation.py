import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from glob import glob

# Define the augmentation sequence using Keras ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

def augment_images(image_dir, save_to_dir, num_augmented_images=5):
    image_paths = glob(os.path.join(image_dir, '*.jpg'))
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = np.expand_dims(image, 0)  # Keras requires a 4D array
        
        # Create a flow of augmented images
        i = 0
        for batch in datagen.flow(image, batch_size=1, save_to_dir=save_to_dir, save_prefix='aug', save_format='jpg'):
            i += 1
            if i >= num_augmented_images:  # Limit the number of augmented images per original image
                break

# Apply augmentation on each shape folder
shapes = ['rectangle', 'ellipse', 'rounded_rectangle']
for shape in shapes:
    image_dir = f'shapes_dataset/{shape}'
    augment_images(image_dir, image_dir)  # Save augmented images in the same directory
