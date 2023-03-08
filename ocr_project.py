import easyocr as ocr
import streamlit as st
from PIL import Image
import numpy as np
import mysql.connector
import pandas as pd
import cv2

db = mysql.connector.connect(
  host="localhost",
  user="root",
  password="your password",
  database="business_cards"
)

cursor = db.cursor()

cursor.execute("CREATE TABLE IF NOT EXISTS ocr_results (id INT AUTO_INCREMENT PRIMARY KEY, image_name VARCHAR(255), result_text TEXT)")

st.title("BizCardX: Extracting Business Card Data Using OCR")
image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])

@st.cache_data 
def load_model(): 
    reader = ocr.Reader(['en'],model_storage_directory='.')
    return reader 

reader = load_model() #load model

if image is not None:

    input_image = Image.open(image) #read image
    img_np = np.array(input_image)
    resized = cv2.resize(img_np, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    result = reader.readtext(threshold)
    result_text = [] #empty list for results
    st.image(input_image) #display image
    with st.spinner("🤖 AI is at Work! "):
        result = reader.readtext(np.array(input_image))
        result_text = [] #empty list for results
        for text in result:
            result_text.append(text[1])
        st.table({"Text": result_text})
        image_name = image.name
        result_text_str = ", ".join(result_text)
        query = "INSERT INTO ocr_results (image_name, result_text) VALUES (%s, %s)"
        values = (image_name, result_text_str)
        cursor.execute(query, values)
        db.commit()
    st.balloons()
else:
    st.write("Upload an Image")
st.markdown("## Previously Extracted Information")

cursor.execute("SELECT * FROM ocr_results")
results = cursor.fetchall()

if len(results) > 0:
    for result in results:
        st.write(f"Image Name: {result[1]}")
        st.write(f"Result Text: {result[2]}")
        st.write("---")

st.markdown("## Delete Extracted Information")
result_to_delete = st.selectbox("Select result to delete", [result[2] for result in results])
if st.button("Delete"):
    cursor.execute(f"DELETE FROM ocr_results WHERE result_text = '{result_to_delete}'")
    db.commit()
    st.write(f"Result '{result_to_delete}' deleted successfully.")

st.markdown("## Display the table")    
query = 'select * from ocr_results'
df = pd.read_sql(query, db)
st.dataframe(df)
