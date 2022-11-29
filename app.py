import pandas as pd
from Plot import predict
import streamlit as st
st.set_page_config(layout="wide")
title = '<p style="font-family: Arial, Helvetica, sans-serif;text-align:center; font-size: 50px;color:red;text-shadow: 2px 2px #080000;">STOCK PRICE PREDICTION </p>'
st.markdown(title, unsafe_allow_html=True)
page_bg_img = """
<style>
.stApp {
background-image: url("https://financialtribune.com/sites/default/files/field/image/17january/04_wb_1242-ab.jpg");
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
hide_style="""
<style>
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
</style>
"""
st.markdown(hide_style,unsafe_allow_html=True)
new_title = '<p style="font-family:sans-serif; font-size: 20px;text-align: left;">Stock market prediction is the act of trying to determine the future value of a company stock or other financial instrument traded on an exchange. The successful prediction of a stocks future price could yield significant profit. In this application we will predict the future closing prices of the stocks by plotting a graph. The volatile nature of stocks makes it nearly impossible to predict the exact stock values but trends can be predicted using technical analysis which is what I am using in this application. Though these values might not be exact but these can help stock brokers during prediction. </p>'
content1 = '<br><p style="font-family:sans-serif; font-size: 20px;">In this application you can check the stock prices of already existing companies from National Stock Exchange or you can upload your company\'s historical data . </p>'
content = '<br><p style="font-family:sans-serif; font-size: 20px">Upload only a CSV file with the following columns </p>'
bullets="""
<ul style="list-style-type:disc">
  <li>Date</li>
  <li>Close</li>
  <li>Open</li>
  <li>High</li>
  <li>Low</li>
  <li>Volume</li>
</ul>
"""
st.markdown(new_title, unsafe_allow_html=True)
st.markdown(content1, unsafe_allow_html=True)
content2 = '<br><p style="font-family:sans-serif; font-size: 20px">Do you want to check the stocks of the following companies (Click \'Submit\' after selecting) </p>'
st.markdown(content2, unsafe_allow_html=True)
option=st.selectbox('',('Select the Company','Apollo','Berge Paint','Nestle India','TCS','HDFC','Indus Tower','Infosys','Indian Oil Corporation'))
content3 = '<br><p style="font-family:sans-serif; font-size: 30px;text-align:center;color:red;">or</p>'
st.markdown(content3, unsafe_allow_html=True)
st.markdown(content, unsafe_allow_html=True)
st.markdown(bullets,unsafe_allow_html=True)
def callback():
         st.session_state.button_clicked =True
spectra = st.file_uploader("Upload a CSV file", type={"csv"})
if spectra is not None:
    df_test = pd.read_csv(spectra)
    a=df_test.columns.tolist()
    if 'Close' in a:
        df_test.rename(columns = {'Close':'close'}, inplace = True)
    if 'Date' in a:
        df_test.rename(columns = {'Date':'date'}, inplace = True)
else:
    if option=='Apollo':
        df_test=pd.read_csv("Data/APOLLOHOSP.csv")
    elif option=='Berge Paint':
        df_test=pd.read_csv("Data/BERGEPAINT.csv")
    elif option=='Nestle India':
        df_test=pd.read_csv("Data/NESTLEIND.csv")
    elif option=="TCS":
        df_test=pd.read_csv("Data/TCS.csv")
    elif option=="HDFC":
        df_test=pd.read_csv("Data/HDFC.csv")
    elif option=="Indus Tower":
        df_test=pd.read_csv("Data/INDUSTOWER.csv")
    elif option=="Infosys":
        df_test=pd.read_csv("Data/INFY.csv")
    elif option=="Indian Oil Corporation":
        df_test=pd.read_csv("Data/IOC.csv")
        
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

m = st.markdown("""<style> div.stButton > button:first-child { background-color: black;border: 2px solid red;color:red;box-shadow: 0px 8px 15px rgba(225, 25, 25, 0.8);}</style>""", unsafe_allow_html=True)
                 
if(st.button('Submit' , on_click=callback)or st.session_state.button_clicked):
    content2 = '<br><p style="font-family:sans-serif; font-size: 20px;">The data passed for prediction </p>'
    st.markdown(content2, unsafe_allow_html=True)         
    st.write(df_test)
    
    content3 = '<br><p style="font-family:sans-serif; font-size: 30px;text-align:center;">Enter the number of days you wish to see the forecast for </p>'
    st.markdown(content3, unsafe_allow_html=True)
    number = st.number_input("")
    m = st.markdown("""<style> div.stButton > button:first-child { text-align:center;background-color: black;border: 2px solid red;color:red;box-shadow: 0px 8px 15px rgba(225, 25, 25, 0.8)}</style>""", unsafe_allow_html=True)
    col1, col2, col3 , col4, col5,col6,col7 = st.columns(7)
    number=int(number)
    
    if col4.button('Check Prediction'):
        predict(df_test,number)
