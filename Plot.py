import streamlit as st
from plotly.offline import plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from datetime import datetime
import plotly.graph_objs as go
from itertools import cycle
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import streamlit.components.v1 as components
import tensorflow as tf
import plotly.io as pio
loaded_model= tf.keras.models.load_model('models/model.h5')
def predict(df_test,number):
  
    df_test['date'] = pd.to_datetime(df_test['date'])
    df_test['Date'] = [d.date() for d in df_test['date']]
    df_test['Time'] = [d.time() for d in df_test['date']]
    df_test['Date'] = pd.to_datetime(df_test['Date'])
    df_pred=df_test.copy()
    df_pred = df_pred[['close']]
    scaler=MinMaxScaler(feature_range=(0,1))
    df_pred=scaler.fit_transform(np.array(df_pred).reshape(-1,1))
    training_size=int(len(df_pred)*0.70)
    test_size=len(df_pred)-training_size
    train_data,test_data=df_pred[0:training_size,:],df_pred[training_size:len(df_pred),:1]
    title1 = '<br><p style="font-family: Arial, Helvetica, sans-serif;background-color: black;border: 2px solid red;box-shadow: 0px 8px 15px rgba(225, 25, 25, 0.8);text-align:center; font-size: 30px;color:red;text-shadow: 2px 2px #080000;">FORECAST</p><br>'
    st.markdown(title1, unsafe_allow_html=True)
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]    
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    train_predict=loaded_model.predict(X_train)
    test_predict=loaded_model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 
    look_back=time_step
    trainPredictPlot = np.empty_like(df_pred)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    testPredictPlot = np.empty_like(df_pred)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df_pred)-1, :] = test_predict
    names = cycle(['Original close price','Train predicted close price','Test predicted close price'])
    plotdf = pd.DataFrame({'date': df_test['Date'],
                           'original_close': df_test['close'],
                          'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                          'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})
    fig,ax=plt.subplots(1,1)
    
    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                              plotdf['test_predicted_close']],
                  labels={'value':'Stock price','date': 'Date'},width=1000,height=800)
    
    fig.update_layout(title_text='',
                      plot_bgcolor='rgba(0,0,0,0.9)', paper_bgcolor='rgba(0,0,0,0.2)', font_size=18, font_color='white', legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    
    with st.container():
        st.plotly_chart(fig,use_container_width=True)
    
    title = '<br><p style="font-family: Arial, Helvetica, sans-serif;background-color: black;border: 2px solid red;box-shadow: 0px 8px 15px rgba(225, 25, 25, 0.8);text-align:center; font-size: 30px;color:red;text-shadow: 2px 2px #080000;">PREDICTED STOCK PRICES</p><br>'
    st.markdown(title, unsafe_allow_html=True)
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    from numpy import array
    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = number
    while(i<pred_days):
    
        if(len(temp_input)>time_step):
        
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
        
            yhat = loaded_model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
       
            lst_output.extend(yhat.tolist())
            i=i+1
        
        else:
        
            x_input = x_input.reshape((1, n_steps,1))
            yhat = loaded_model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
        
            lst_output.extend(yhat.tolist())
            i=i+1
    last_days=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+pred_days+1)
    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]

    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat

    last_original_days_value[0:time_step+1] = scaler.inverse_transform(df_pred[len(df_pred)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    new_pred_plot = pd.DataFrame({
        'last_original_days_value':last_original_days_value,
        'next_predicted_days_value':next_predicted_days_value
    })
    num=str(number)
    names = cycle(['Last 15 days close price','Predicted next {} days close price'.format(num)])

    fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                          new_pred_plot['next_predicted_days_value']],
                  labels={'value': 'Stock price','index': 'Days'},width=1000,height=800)
    
    fig.update_layout(title_text='Comparison between last 15 days vs next {} days'.format(num),
                      plot_bgcolor='rgba(0,0,0,0.9)', paper_bgcolor='rgba(0,0,0,0.2)', font_size=18, font_color='white',legend_title_text='Close Price')

    fig.update_traces(line_color="#E3022F")
    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_yaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)
    with st.container():
        st.plotly_chart(fig,use_container_width=True)