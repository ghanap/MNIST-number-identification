import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 1. Load ONLY the CNN model (No PCA or Scaler needed!)
@st.cache_resource
def load_assets():
    # Make sure this file exists in your repository
    model = tf.keras.models.load_model('mnist_cnn.h5')
    return model

model = load_assets()

st.title("Handwritten Digit Classifier (CNN)")
st.write("Draw a digit (0-9) in the box below!")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # 1. Process drawing: Convert to grayscale and resize to 28x28
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    img_28x28 = img.resize((28, 28))
    
    # 2. Convert to array and normalize to 0-1 range
    img_array = np.array(img_28x28).astype('float32') / 255.0
    
    # Check if user drew something
    if np.max(img_array) > 0:
        # 3. Reshape for CNN: (1, 28, 28, 1)
        # This adds the 'Batch' and 'Channel' dimensions
        img_input = img_array.reshape(1, 28, 28, 1)
        
        # 4. Predict
        predictions = model.predict(img_input)
        final_prediction = np.argmax(predictions)
        
        st.header(f"Prediction: {final_prediction}")
        st.write(f"The model is {np.max(predictions)*100:.2f}% confident.")
        
        # Optional: Show what the model sees
        with st.expander("Debug View"):
            st.image(img_28x28, width=150)
    else:
        st.info("Please draw a digit to see the prediction.")