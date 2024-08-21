import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import json
import zipfile
import io
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import Zeros, Ones

# Define model directories
model_directories = {
    'Model 1': 'custom_model.h5',
    #'Model 2': 'transfer_model.h5'
    'Model 2': 'model_2.h5'
}

#tf.keras.utils.get_custom_objects().update({
#    'BatchNormalization': BatchNormalization,
#    'Zeros': Zeros,
#    'Ones': Ones
#})

# Load the saved model based on the selected model type
#@st.cache(allow_output_mutation=True)
@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    custom_objects = {'BatchNormalization': BatchNormalization, 'Zeros': Zeros, 'Ones': Ones}
    #model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    model = tf.keras.models.load_model(model_path)
    return model

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to a NumPy array
    image = np.array(image)
    # Resize the image
    image = cv2.resize(image, (256, 256))
    # Normalize the image
    #image = image / 255.0
    # Expand dimensions to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    return image

# Load categories and example images from JSON file
def load_categories(json_path):
    with open(json_path, 'r') as f:
        categories = json.load(f)
    return categories['classes']

# Extract image from zip file
def extract_image_from_zip(zip_path, file_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(file_path) as file:
            image = Image.open(io.BytesIO(file.read()))
    return image

# Streamlit app
def main():
    model1 = load_model(model_directories['Model 1'])
    model2 = load_model(model_directories['Model 2'])

    models = {
        "Custom Model": model1,
        "Transfer Model": model2
    }
    
    st.title('Yoga Pose Identification')
    
    # Dropdown to select model:
    model_name = st.selectbox(
        'Which model would you like to use?',
        list(models.keys()))

    chosen_model = models[model_name]

    st.write('You selected:', model_name)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        image = Image.open(uploaded_file)



        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make predictions
        predictions = chosen_model.predict(processed_image)
        
        # Convert class probabilities to a single prediction of the most likely class
        most_likely_class = np.argmax(predictions, axis=1)
        
        # Load categories and example images
        poses = load_categories('class_info.json')
        
        # Get the class name for the most likely class (for lighter version without example photos)
        #class_name = next((pose['class_name'] for pose in poses if pose['class_number'] == most_likely_class), "Unknown")
        #st.write(f"Predicted Pose: {class_name}")

        # Find the class name and file path corresponding to the most likely class
        class_info = next((pose for pose in poses if pose['class_number'] == most_likely_class), None)
        if class_info:
            class_name = class_info['class_name']
            file_path = class_info['file_path']
            
            st.write(f"Predicted Pose: {class_name}")
            
            # Extract and display the example image
            example_image = extract_image_from_zip('examples.zip', "content/"+file_path)
            st.image(example_image, caption=f'Example of {class_name}', use_column_width=True)
        else:
            st.write("Unknown class")

if __name__ == "__main__":
    main()