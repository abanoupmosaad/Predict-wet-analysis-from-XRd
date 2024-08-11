# importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsRegressor
import pickle

# configure page
st.set_page_config(layout='wide') 


# loading data
df = pd.read_excel("Wet Analysis V XRD.xlsx", encoding="ISO-8859-1")

# sidebar
option = st.sidebar.selectbox("Pick a choice:",['Home'])

if option == 'Home':
    st.title("Predict Wet Analaysis From XRD")
    st.text('Author: @Abanoup Mosaad')
    #st.dataframe(df.head())
    Gaunge = st.number_input("Enter Gaunge")
    TFeXR = st.number_input("Enter TFe XR")
    MFeXR = st.number_input("Enter MFe XR")
    MetnXR = st.number_input("Enter Metn XR")
    TfeEXO = st.number_input("Enter TFE EXO")
    FemEXO = st.number_input("Enter Fem EXO")
    MetnEXO = st.number_input("Enter Metn EXO")
    FeoEXO = st.number_input("Enter Feo EXO")
    btn = st.button("Submit")
    # df.drop(columns=['Names','emails','Country'], inplace=True)
    # X = df.drop("Clicked", axis=1)
    # y = df['Clicked']
    # ms = MinMaxScaler()
    # X = ms.fit_transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # clf = LogisticRegression()
    # clf.fit(X_train, y_train)
    # ms = MinMaxScaler()
    GetmetWet = pickle.load(open('Get met Wet Analysis From XRD.pkl','rb'))
    GetfemWet = pickle.load(open('Get fem Wet Analysis From XRD.pkl','rb'))
GetmetWetresult = GetmetWet.predict([[Gaunge, TFeXR, MFeXR, MetnXR, TfeEXO, FemEXO, MetnEXO, FeoEXO]])
GetfemWetresult = GetfemWet.predict([[Gaunge, TFeXR, MFeXR, MetnXR, TfeEXO, FemEXO, MetnEXO, FeoEXO]])

# Calculate TfeWet result
if GetmetWetresult != 0:  # Avoid division by zero
    GetTfeWetresult = (GetfemWetresult / GetmetWetresult) * 100
else:
    GetTfeWetresult = float('inf')  # Handle division by zero

# Streamlit button to trigger the display
if st.button('Show Results'):
    st.write(f"Tfe% = {GetTfeWetresult:.2f}")
    st.write(f"Fem% = {GetfemWetresult[0]:.2f}")
    st.write(f"Met% = {GetmetWetresult[0]:.2f}")
#elif option == 'EDA':
    #st.title("Facebook EDA")

    #col1, col2 = st.columns(2)

    #fig = px.scatter(data_frame=df, x='Time Spent on Site', y='Salary', color='Clicked')
   # st.plotly_chart(fig)

    #with col1:
        #fig = px.violin(data_frame=df, x='Time Spent on Site')
       # st.plotly_chart(fig)

    #with col2:
        #df['Clicked'].astype('O')
        #fig = px.bar(data_frame=df, x=df['Clicked'])
        #st.plotly_chart(fig)       

        # fig = plt.figure()
        # df['Clicked'].value_counts().plot(kind='bar')
        # st.pyplot(fig)

#elif option == "ML":
   # st.title("Ads Clicked Prediction")
   # st.text("In this app, we will predict the ads click using salary and time spent on website")
    #st.text("Please enter the following values:")

# building model
    #Gaunge = st.number_input("Enter Gaunge")
    #TFeXR = st.number_input("Enter TFe XR")
    #MFeXR = st.number_input("Enter MFe XR")
    #MetnXR = st.number_input("Enter Metn XR")
    #TfeEXO = st.number_input("Enter TFE EXO")
    #FemEXO = st.number_input("Enter Fem EXO")
    #MetnEXO = st.number_input("Enter Metn EXO")
    #FeoEXO = st.number_input("Enter Feo EXO")
   # btn = st.button("Submit")
    # df.drop(columns=['Names','emails','Country'], inplace=True)
    # X = df.drop("Clicked", axis=1)
    # y = df['Clicked']
    # ms = MinMaxScaler()
    # X = ms.fit_transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # clf = LogisticRegression()
    # clf.fit(X_train, y_train)
    # ms = MinMaxScaler()
   # clf = pickle.load(open('my_model.pkl','rb'))
    #result = clf.predict([[Gaunge, TFeXR,MFeXR,MetnXR,TfeEXO,FemEXO,MetnEXO,FeoEXO]])

    #if btn:
            #st.write(result)

