import streamlit as st 
import cv2
import numpy as np
from PIL import Image


#Load the Haar cascade for face detection
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Title and Instruction
st.title("Face Detection App")
st.write("""
### Instructions:
1. Upload an image with faces.
2. Adjust the detection parameter ( minNeighbors and scaleFactor) using the sliders.
3. Choose the rectangle color for highlighting detected faces.
4. Save the image with detected faces if desired.
         """)

#Uploading the file 
uploaded_file = st.file_uploader("Upload an image", type = ["jpg", "jpeg", "png"])

#Parameters for face detection
scale_factor = st.slider("Adjust scaleFactor (How much the image size is reduced at each image scale)", 1.1,2.0,1.1,0.1)
min_neighbors = st.slider("Adjust minNeighbors (How many neighbors each rectangle should have to retain it)", 3, 10, 5, 1)


# Color picker for rectangle
rectangle_color = st.color_picker("Pick a rectangle color", "#FF0000")
rectangle_bgr = tuple(int(rectangle_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))  # Convert hex to BGR


if uploaded_file is not None:
    # Load image
    image = np.array(Image.open(uploaded_file))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)


    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), rectangle_bgr, 2)

    # Display the image with detected faces
    st.image(image, caption="Image with Detected Faces", use_column_width=True, channels="BGR")

    # Option to save the image
    if st.button("Save Image"):
        save_path = "detected_faces.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        st.success(f"Image saved as {save_path}")



