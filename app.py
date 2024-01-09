import cv2
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="عالم النباتات",
    page_icon="logo_ world of plants.jpeg",
    # layout="wide",
    initial_sidebar_state="expanded"
)
st.image("logo_ world of plants.jpeg", width=100)
st.markdown(
    f"""
    <style>
        .custom-btn {{
            background-color: #379237;
            color: #F7F7F7;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }}
        .custom-btn:hover {{
            background-color: #FFB534;
        }}
        .custom-btn:active {{
            background-color: #379237;
        }}
    </style>
    <a href='https://c789ba-4.myshopify.com/' class='custom-btn' style='color: #F7F7F7; text-decoration: none;' >الرجوع الى الموقع</a>
    """,
    unsafe_allow_html=True
)


st.title("مرحبا بك في عالم النباتات نساعدك لمعرفك الآفات الزراعية على محصول البرتقال ")
# image = Image.open("images (7).jpg")
# resized_image = image.resize((500, 300))
# st.image(resized_image, caption='Resized Image')

# st.image('images (7).jpg', caption='Sunrise by the mountains')

import streamlit as st
from PIL import Image

# Load and resize the first image
image1 = Image.open("images (7).jpg")
resized_image1 = image1.resize((500, 300))

# Load and resize the second image
image2 = Image.open("OrangeVariousBenefits2.jpg")
resized_image2 = image2.resize((500, 300))

# Create a two-column layout
col1, col2 = st.columns(2)

# Display the resized image in the first column
col1.image(resized_image1, caption='  ورقة البرتقال')

# Display the resized second image in the second column
col2.image(resized_image2, caption='ثمرة البرتقال')



class_names = [
    'Black spot',
    'canker',
    'Greening',
    'Melanose',
]



# from st_pages import Page, Section, show_pages, add_page_title

model = keras.models.load_model('keras_model.h5')
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize the image to match the input size of the model
    img = img.convert('RGB')  # Convert image to RGB format
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return keras.applications.mobilenet.preprocess_input(img_array)

def main():
    # Set page width and center content
    max_width = 1000
    st.markdown(
        f"""
        <style>
        .reportview-container .main .block-container{{
            max-width: {max_width}px;
            padding-top: 1rem;
            padding-bottom: 1rem;
            margin: 0 auto;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # st.title("Image Classification")
    # st.subheader("Identifying plant diseases using artificial intelligence")

    st.subheader("رفع صورة ورقة البرتقال لتحديد المرض")
    st.subheader("مرحبًا! هل لديك شكوك حول حالة ورقة البرتقال؟ انقر على الزر أدناه لرفع صورة ورقة البرتقال، وسنستخدم التكنولوجيا الحديثة والذكاء الاصطناعي لتحليل الصورة وتقديم تشخيص دقيق لأي مشكلة قد تكون موجودة.")
    st.text("كيفية الاستفادة:")
    st.text("1. انقر على الزر رفع صورة")
    st.text("2. حدد ورقة التفاح بوضوح في الصورة.")
    st.text("3. انتظر لحظات حتى يتم تحليل الصورة.")
    st.text("4. ستتلقى تقريرًا يحدد المشكلة ويقدم توصيات إن وجدت.")
    st.text("ابدأ الآن في فحص ورقة التفاح الخاصة بك!")
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png']) 
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make predictions
        predictions = model.predict(processed_image)
        predicted_class_index = predictions.argmax()
        predicted_class_name = class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100

        # Display predicted class and confidence
        st.markdown(
            """
            <div style='text-align: center;'>
                <h2 style='font-weight: bold; color: #0072B2;'>Predicted Class</h2>
                <h3 style='font-weight: bold; color: #0072B2;'>{}</h3>
            </div>
            """.format(predicted_class_name),
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style='text-align: center;'>
                <h2 style='font-weight: bold; color: #0072B2;'>Confidence</h2>
                <h3 style='font-weight: bold; color: #0072B2;'>{:.2f}%</h3>
            </div>
            """.format(confidence),
            unsafe_allow_html=True
        )

        # Display other classes
        st.markdown(
            """
            <div style='text-align: center;'>
                <h2 style='font-weight: bold; color: #0072B2;'>Other Classes</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        for i, class_name in enumerate(class_names):
            if i != predicted_class_index:
                st.markdown(
                    """
                    <div style='text-align: center;'>
                        <h3 style='font-weight: bold; color: #0072B2;'>{}: {:.2f}%</h3>
                    </div>
                    """.format(class_name, predictions[0][i] * 100),
                    unsafe_allow_html=True
                )


if __name__ == '__main__':
    main()
