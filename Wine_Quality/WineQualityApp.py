import pickle 
import pandas as pd
import streamlit as st
#upload  the data 
DataModelleing =pickle.load(open(r'/Users/moussaelrahmoun/Desktop/Dat158Project/Wine_Quality/WienQuality_prediction.sav','rb'))
st.title('wine_qualti')
st.sidebar.header('feature selection ')
fixed_acidity = st.number_input("Fixed Acidity")
volatile_acidity = st.number_input("Volatile Acidity")
citric_acid = st.number_input("Citric Acid")
residual_sugar = st.number_input("Residual Sugar")
chlorides = st.number_input("Chlorides")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide")
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide")
density = st.number_input("Density")
pH = st.number_input("pH")
sulphates = st.number_input("Sulphates")
alcohol = st.number_input("Alcohol")

# datafram is neded to use savd data from the input 
Data_df=Data_df = pd.DataFrame({
    'fixed acidity': [fixed_acidity],
    'volatile acidity': [volatile_acidity],
    'citric acid': [citric_acid],
    'residual sugar': [residual_sugar],
    'chlorides': [chlorides],
    'free sulfur dioxide': [free_sulfur_dioxide],
    'total sulfur dioxide': [total_sulfur_dioxide],
    'density': [density],
    'pH': [pH],
    'sulphates': [sulphates],
    'alcohol': [alcohol]
}, index=[0])


predict =st.sidebar.button('predicrt')
if predict:
    result=DataModelleing.predict(Data_df)             # using our saved rf_model to predict the input from the user 
    if result==1:
        st.sidebar.write('God  Quality')

    else:
        st.sidebar.write("Bad  Quality ")