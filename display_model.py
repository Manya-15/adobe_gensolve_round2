import cv2
import numpy as np
from keras.models import load_model

# Parameters
img_size = 200
dimData = img_size * img_size

# Load the trained model
model = load_model('shapes_model.h5')

# Preprocess the input image
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_resized = img_resized.astype('float32')
    img_resized /= 255
    img_resized = img_resized.reshape(1, dimData)
    return img_resized

# Predict the shape in the image
def predict_shape(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)[0].tolist()
    
    # Mapping prediction to shape
    shapes = ['triangle', 'star', 'square', 'circle', 'rectangle', 'ellipse', 'rounded_rectangle']
    max_index = np.argmax(prediction)
    return shapes[max_index], prediction[max_index]

# Test with an image
image_path = 'test1.jpg'
shape, confidence = predict_shape(image_path)
print(f"The predicted shape is: {shape} with confidence: {confidence:.2f}")
