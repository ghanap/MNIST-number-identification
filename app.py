import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the saved components
@st.cache_resource
def load_assets():
    model = joblib.load('mnist_model.pkl')
    pca = joblib.load('pca_transformer.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, pca, scaler

model, pca, scaler = load_assets()

st.title("Handwritten Digit Classifier")
st.write("Draw a digit (0-9) in the box below!")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  # Fixed fill color
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Process the drawing
if canvas_result.image_data is not None:
    # 1. Convert canvas image to grayscale and resize to 28x28
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    img_28x28 = img.resize((28, 28))
    
    # 2. Flatten for the model (1 row, 784 columns)
    img_array = np.array(img_28x28).reshape(1, -1)
    
    # Check if the user has actually drawn something (avoid predicting empty canvas)
    if np.max(img_array) > 0:
        # 3. Apply PCA
        img_pca = pca.transform(img_array)
        
        # 4. Apply Scaler
        img_scaled = scaler.transform(img_pca)
        
        # 5. Predict
        prediction = model.predict(img_scaled)
        
        st.header(f"Prediction: {prediction[0]}")
        st.write(f"The model sees a {prediction[0]}")
    else:
        st.info("Please draw a digit to see the prediction.")