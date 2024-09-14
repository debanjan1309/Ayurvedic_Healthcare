from PIL import Image
import os

main_data_dir = "F:\project\SIH23\classification\Segmented Medicinal Leaf Images"

# Load the input image
input_image_path = "WhatsApp Image 2023-09-15 at 2.01.34 PM.jpeg"
input_image = Image.open(input_image_path)

# Define the target size (1600x1200)
target_size = (1600, 1200)

# Resize the image
resized_image = input_image.resize(target_size)

# Save the resized image
output_image_path = "image.jpg"
resized_image.save(output_image_path)

# Optional: Display the resized image
# resized_image.show()

# Test Image
image_path = "image.jpg"

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load and display the image
img = mpimg.imread(image_path)

# plt.imshow(img)
# plt.axis('off')  # Turn off axis labels and ticks
# plt.show()

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
model = tf.keras.models.load_model('plant_identification_model2.h5')

# Load and preprocess the image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input(image_array)
    return preprocessed_image

# Perform prediction
def predict_plant(image_path, label_mapping):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    
    # Map model's numeric predictions to labels
    predicted_label_index = np.argmax(predictions)
    predicted_label = label_mapping[predicted_label_index]
    confidence = predictions[0][predicted_label_index]
    
    return predicted_label, confidence

# Create label mapping based on subdirectory names
label_mapping = {i: label for i, label in enumerate(sorted(os.listdir(main_data_dir)))}

# Provide the path to the image you want to classify
predicted_label, confidence = predict_plant(image_path, label_mapping)

# Print the prediction
print(f"Predicted Label: {predicted_label}")
print(f"Confidence: {confidence:.2f}")

