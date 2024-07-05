import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
import streamlit as st

# Constants
IMAGE_SIZE = 128

# Load model and class indices
model = load_model('./model/skin_cancer_model.h5')

with open('./model/class_indices.json', 'r') as json_file:
    class_indices = json.load(json_file)

cancer_descriptions = {
    'akiec': 'Actinic keratoses and intraepithelial carcinoma / Bowen\'s disease',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas, and hemorrhage)',
    'healthy': 'This is healthy skin'
}

def generate_grid_image(image_array, grid_size=30):
    grid = np.zeros((grid_size, grid_size))
    image_shape = image_array.shape[:2]
    step_x = image_shape[0] // grid_size
    step_y = image_shape[1] // grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            grid[i, j] = np.mean(image_array[i*step_x:(i+1)*step_x, j*step_y:(j+1)*step_y])

    return grid

def generate_numerical_grid_image(image_array, threshold=0.5, grid_size=30):
    grid = np.zeros((grid_size, grid_size))
    image_shape = image_array.shape[:2]
    step_x = image_shape[0] // grid_size
    step_y = image_shape[1] // grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            mean_value = np.mean(image_array[i*step_x:(i+1)*step_x, j*step_y:(j+1)*step_y])
            grid[i, j] = 1 if mean_value >= threshold else 0

    return grid

def generate_bounding_box(image_array, threshold=0.5):
    img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        confidence = 1 if np.mean(image_array[y:y+h, x:x+w]) >= threshold else 0
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255 * confidence, 0), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def generate_confidence_grid(predictions, grid_size):
    confidence_grid = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            confidence_grid[i, j] = predictions[0, i * grid_size + j] if i * grid_size + j < predictions.shape[1] else 0
    return confidence_grid

def predict_cancer(image_path):
    img = keras_image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = list(class_indices.keys())[predicted_index]
    confidence = predictions[0][predicted_index]

    if predicted_label == 'healthy':
        return predicted_label, confidence, None, None, None, None, img_array

    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    grid_image = generate_grid_image(img_array[0])
    numerical_grid_image = generate_numerical_grid_image(img_array[0])
    bounding_box_image = generate_bounding_box(img)
    confidence_grid = generate_confidence_grid(predictions, grid_size=numerical_grid_image.shape[0])

    return predicted_label, confidence, grid_image, numerical_grid_image, bounding_box_image, confidence_grid, img

st.title("Skin Cancer Prediction")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image_path = f"./{uploaded_file.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    predicted_label, confidence, grid_image, numerical_grid_image, bounding_box_image, confidence_grid, original_image_array = predict_cancer(image_path)

    st.write(f"Predicted cancer type: {predicted_label}")
    st.write(f"Confidence: {confidence}")
    if original_image_array is not None:
        st.write(f"Description: {cancer_descriptions[predicted_label]}")
        
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(grid_image, cmap='hot', interpolation='nearest')
        axs[0].set_title('30x30 Grid with Markings')
        axs[1].imshow(numerical_grid_image, cmap='binary', interpolation='nearest')
        axs[1].set_title('30x30 Grid Mask')
        axs[2].imshow(cv2.cvtColor(original_image_array, cv2.COLOR_BGR2RGB))
        axs[2].set_title('Original Image')
        axs[3].imshow(bounding_box_image)
        axs[3].set_title('Bounding Marking Area')
        
        st.pyplot(fig)
