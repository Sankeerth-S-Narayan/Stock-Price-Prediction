import pandas as pd
from Plot import predict
import streamlit as st
title = '<p style="font-family: Arial, Helvetica, sans-serif; font-size: 70px;text-align: center;color:red;text-shadow: 2px 2px #080000;">Stock Price Prediction </p>'
st.markdown(title, unsafe_allow_html=True)

page_bg_img = """
<style>
.stApp {
background-image: url("https://axisfinancialgroup.net/wp-content/uploads/2018/03/SLIDE-1.jpg");
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
new_title = '<p style="font-family:sans-serif; font-size: 20px;">Stock market prediction is the act of trying to determine the future value of a company stock or other financial instrument traded on an exchange. The successful prediction of a stocks future price could yield significant profit. </p>'
content1 = '<br><p style="font-family:sans-serif; font-size: 20px;">In this application we will predict the future closing prices of the stocks that you have entered  </p>'
content = '<br><p style="font-family:sans-serif; font-size: 20px;">Upload only a CSV file with the following columns </p>'
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

st.markdown(content, unsafe_allow_html=True)
st.markdown(bullets,unsafe_allow_html=True)
def callback():
         st.session_state.button_clicked =True
spectra = st.file_uploader("Upload a CSV file", type={"csv"})
if spectra is not None:
    df_test = pd.read_csv(spectra)
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False
                 
if(st.button('Submit' , on_click=callback)or st.session_state.button_clicked):
    st.write(df_test)
    m = st.markdown("""<style> div.stButton > button:first-child { background-color: black;border: 2px solid red;color:red;box-shadow: 0px 8px 15px rgba(225, 25, 25, 0.8);text-align:center;}</style>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    if col2.button('Check Prediction'):
        predict(df_test)
