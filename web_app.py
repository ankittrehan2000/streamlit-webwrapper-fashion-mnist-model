import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2

model = tf.keras.models.load_model('recognition_model.hdf5')
prediction_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#streamlit code to create a web page
st.write("Fashion Categorization model")
st.write("This is a web page that leverages a simple classification model to predict a fashion category")
imgFile = st.file_uploader("Please upload an image file", type=["jpg", "png"])

#resize image and run through the model
def import_and_predict(image_data, model):
  size=(360, 360)
  image_data=image_data.resize(size)
  img=ImageOps.grayscale(image)
  img=img.resize((28,28))
  img=np.expand_dims(img,0)
  img=(img/255.0)
                  
  img=1-img
  
  prediction = model.predict(img)
  return prediction
  

if imgFile is None:
  st.text("Please upload an image")
else:
  image = Image.open(imgFile)
  st.image(image, use_column_width=True)
  prediction = import_and_predict(image, prediction_model)
  category = np.argmax(prediction)
  object_type = class_names[category]
  st.write("The given image is a " + object_type)
