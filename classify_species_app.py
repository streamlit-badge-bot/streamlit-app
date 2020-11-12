import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# A simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 5.8, 4.9, 4.5)
    sepal_width = st.sidebar.slider('Sepal width', 5.0, 3.4, 6.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 5.9, 4.3)
    petal_width = st.sidebar.slider('Petal width', 0.4, 2.8, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)


prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction Results')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability of Species')
st.write(prediction_proba)

video_file = open('classify_species_app_record_12_11_2020.mp4', 'rb')
video_bytes = video_file.read()

st.subheader('Screencast Recording')
st.video(video_bytes)
