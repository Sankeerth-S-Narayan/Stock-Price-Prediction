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
    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                              plotdf['test_predicted_close']],
                  labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    plot(fig, auto_open=True)  
    pio.write_html(fig, file='index.html', auto_open=True)

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
        #print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
        
            yhat = loaded_model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
        #print(temp_input)
       
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
                  labels={'value': 'Stock price','index': 'Days'})
    fig.update_layout(title_text='Compare last 15 days vs next {} days'.format(num),
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    plot(fig, auto_open=True)
    pio.write_html(fig, file='index.html', auto_open=True)