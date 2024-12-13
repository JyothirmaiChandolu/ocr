#!  C:\Users\chand\python\myenv\Scripts\python.exe

import streamlit as st
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
import pandas as pd  # Import pandas for table display

# Path to your manually downloaded models
det_model_dir = './inference/det_r50_vd_db_infer'  # Detection model
rec_model_dir = './inference/rec_r34_vd_crnn_en_infer'  # Recognition model
cls_model_dir = './inference/cls_mv3_infer'  # Classification model

# Initialize PaddleOCR with all inferences
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    det_model_dir=det_model_dir,
    rec_model_dir=rec_model_dir,
    cls_model_dir=cls_model_dir,
)

# Streamlit UI for image upload
st.title("Team-6 OCR App")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the uploaded file to a NumPy array for PaddleOCR
    img_array = np.array(image)[:, :, ::-1]  # Convert RGB to BGR

    # Perform OCR using the specified models
    result = ocr.ocr(img_array, cls=True)

    # Prepare data for the table
    table_data = []  # List to store table rows
    for idx, line in enumerate(result[0], start=1):  # Iterate through OCR results
        text = line[1][0]  # Detected text
        score = line[1][1]  # Confidence score
        table_data.append({"Serial Number": idx, "Word": text, "Score": score})  # Append to table

    # Convert the data to a pandas DataFrame for display
    df = pd.DataFrame(table_data)

    # Display the table in Streamlit
    st.subheader("OCR Results Table")
    st.dataframe(df)  # Alternatively, use st.table(df)
