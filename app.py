import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("OpenCV operations")
st.subheader("Image operations")

st.write("Upload an image and choose an OpenCV operation")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = np.array(img)

    option = st.selectbox(
        'What color space do you want to convert to?',
         ('RGB', 'HSV', 'Canny edge detection'))
    st.write('You selected:', option)

    if option == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif option == 'Canny edge detection':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.Canny(img, 100, 200)
        
    st.image(img, channels="BGR")
