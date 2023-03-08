import easyocr as ocr  #OCR
import streamlit as st  #Web App
from PIL import Image #Image Processing
import numpy as np #Image Processing 
import mysql.connector
import pandas as pd
import cv2

# Connect to the database
db = mysql.connector.connect(
  host="localhost",
  user="root",
  password="your password",
  database="business_cards"
)

# Get a cursor to execute SQL queries
cursor = db.cursor()

# Create the ocr_results table if it doesn't exist
cursor.execute("CREATE TABLE IF NOT EXISTS ocr_results (id INT AUTO_INCREMENT PRIMARY KEY, image_name VARCHAR(255), result_text TEXT)")

#title
st.title("BizCardX: Extracting Business Card Data with OCR")



#image uploader
image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])

@st.cache_data 
def load_model(): 
    reader = ocr.Reader(['en'],model_storage_directory='.')
    return reader 

reader = load_model() #load model

if image is not None:

    input_image = Image.open(image) #read image
    # Perform image processing techniques to enhance the image quality before passing it to the OCR engine
    # Convert the image to a numpy array
    img_np = np.array(input_image)

    # Apply image processing techniques
    # Resize the image to reduce processing time and improve OCR accuracy
    resized = cv2.resize(img_np, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    # Convert the image to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to binarize the image
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Perform OCR on the processed image
    result = reader.readtext(threshold)

    result_text = [] #empty list for results
    st.image(input_image) #display image

    with st.spinner("🤖 AI is at Work! "):
        

        result = reader.readtext(np.array(input_image))

        result_text = [] #empty list for results


        for text in result:
            result_text.append(text[1])

        

        # Display the extracted information in a table
        st.table({"Text": result_text})

        # Insert the OCR results into the database
        image_name = image.name
        result_text_str = ", ".join(result_text)
        query = "INSERT INTO ocr_results (image_name, result_text) VALUES (%s, %s)"
        values = (image_name, result_text_str)
        cursor.execute(query, values)
        db.commit()

    st.balloons()

else:
    st.write("Upload an Image")

# Add a section to display the OCR results stored in the database
st.markdown("## Previously Extracted Information")

cursor.execute("SELECT * FROM ocr_results")
results = cursor.fetchall()

if len(results) > 0:
    for result in results:
        st.write(f"Image Name: {result[1]}")
        st.write(f"Result Text: {result[2]}")
        st.write("---")

# Add a section to delete OCR results from the database
st.markdown("## Delete Extracted Information")
result_to_delete = st.selectbox("Select result to delete", [result[2] for result in results])
if st.button("Delete"):
    cursor.execute(f"DELETE FROM ocr_results WHERE result_text = '{result_to_delete}'")
    db.commit()
    st.write(f"Result '{result_to_delete}' deleted successfully.")

#to display the data from database
st.markdown("## Display the table")    
query = 'select * from ocr_results'
df = pd.read_sql(query, db)
st.dataframe(df)