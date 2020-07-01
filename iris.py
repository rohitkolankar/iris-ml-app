# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:01:44 2020

@author: Rohit Kolankar
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:55:54 2020

@author: Rohit Kolankar
"""

import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("classifieriris.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_flower(sepalLength,sepalWidth,petalLength,petalWidth):
    
    
    list = [sepalLength,sepalWidth,petalLength,petalWidth]
    new = np.array(list,dtype=np.float32)
    prediction=classifier.predict([new])
    print(prediction)
    return prediction



def main():
    st.title("Iris flower classification")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Iris Flower classification ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    

    # Add a slider to the sidebar:
    
    
    st.sidebar.title("Choose the Parameters")
   
    sepalLength = st.sidebar.slider(
    'Select a range of values for Petal width',
    0.0, 100.0
    )
    sepalWidth = st.sidebar.slider(
    'Select a range of values for Sepal Length',
    0.0, 100.0
    )
    petalLength = st.sidebar.slider(
    'Select a range of values for Sepal Width',
    0.0, 100.0
    )
    
    petalWidth = st.sidebar.slider(
    'Select a range of values for Petal Length',
    0.0, 100.0
    )
    
    
    
    st.write("Sepal Length", sepalLength)
    st.write("Sepal Width", sepalWidth)
    st.write("Petal Length",petalLength )
    st.write("Petal Width", petalWidth)
    result=""
    category = ['Iris-Setosa','iris-Verginica','Iris-Versicolor']
    if st.button("Predict"):
        result=predict_flower(sepalLength,sepalWidth,petalLength,petalWidth)
    if result=='':
        st.success('Please Enter the data')
    elif result==0:
        st.success('Iris-Setosa')
    elif result==1:
        st.success('iris-Verginica')
    else:
        st.success('iris-Versicolor')
    if st.button("About"):
        st.text("Author:Rohit Kolankar")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    