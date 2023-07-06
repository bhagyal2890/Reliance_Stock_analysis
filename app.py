import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import datetime
import streamlit as st
import model_building as m


with st.sidebar:
    st.markdown("# Reliance_Stock_Analysis")
    user_input = st.multiselect('Please select the stock',['RELIANCE.NS'],['RELIANCE.NS'])

    # user_input = st.text_input('Enter Stock Name', "ADANIENT.NS")
    st.markdown("### Choose Date for your anaylsis")
    START = st.date_input("From",datetime.date(2015, 1, 1))
    END = st.date_input("To",datetime.date(2023, 2, 28))
    bt = st.button('Submit') 

#adding a button
if bt:

# Importing dataset------------------------------------------------------
    df = yf.download('RELIANCE.NS', start=START, end=END)
    plotdf, future_predicted_values =m.create_model(df)
    df.reset_index(inplace = True)
    st.title('Reliance Stock Market Prediction')
    st.header("Data We collected from the source")
    st.write(df)

    reliance_1=df.drop(["Adj Close"],axis=1).reset_index(drop=True)
    reliance_2=reliance_1.dropna().reset_index(drop=True)

    reliance=reliance_2.copy()
    reliance['Date']=pd.to_datetime(reliance['Date'],format='%Y-%m-%d')
    reliance=reliance.set_index('Date')
    st.title('EDA')
    st.write(reliance)


# ---------------------------Graphs--------------------------------------

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Visualizations')

    st.header("Graphs")
    plt.figure(figsize=(20,10))
    st.pyplot(fig=plt)
    #Plot 1
    plt.subplot(2,2,1)
    plt.plot(reliance['Open'],color='green')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.title('Open')
    st.pyplot(fig=plt)
    #Plot 2
    plt.subplot(2,2,2)
    plt.plot(reliance['Close'],color='red')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Close')
    st.pyplot(fig=plt)
    #Plot 3
    plt.subplot(2,2,3)
    plt.plot(reliance['High'],color='green')
    plt.xlabel('Date')
    plt.ylabel('High Price')
    plt.title('High')
    st.pyplot(fig=plt)
    #Plot 4
    plt.subplot(2,2,4)
    plt.plot(reliance['Low'],color='red')
    plt.xlabel('Date')
    plt.ylabel('Low Price')
    plt.title('Low')
    st.pyplot(fig=plt)
    
    st.header("Graphs- Box Plot")
    plt.figure(figsize=(20,10))

    # Creating box-plots
    plt.figure(figsize=(20,10))
    #Plot 1
    plt.subplot(2,2,1)
    plt.boxplot(reliance['Open'])
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.title('Open')
    st.pyplot(fig=plt)
    #Plot 2
    plt.subplot(2,2,2)
    plt.boxplot(reliance['Close'])
    plt.xlabel('Date')
    plt.ylabel('Cloes Price')
    plt.title('Close')
    st.pyplot(fig=plt)
    #Plot 3
    plt.subplot(2,2,3)
    plt.boxplot(reliance['High'])    
    plt.xlabel('Date')
    plt.ylabel('High Price')
    plt.title('High')
    st.pyplot(fig=plt)
    #Plot 4
    plt.subplot(2,2,4)
    plt.boxplot(reliance['Low'])
    plt.xlabel('Date')
    plt.ylabel('Low Price')
    plt.title('Low')
    st.pyplot(fig=plt)

    # Ploting Histogram
    plt.figure(figsize=(20,18))
    #Plot 1
    plt.subplot(2,2,1)
    plt.hist(reliance['Open'],bins=50, color='green')
    plt.xlabel("Open Price")
    plt.ylabel("Frequency")
    plt.title('Open')
    st.pyplot(fig=plt)
    #Plot 2
    plt.subplot(2,2,2)
    plt.hist(reliance['Close'],bins=50, color='red')
    plt.xlabel("Close Price")
    plt.ylabel("Frequency")
    plt.title('Close')
    st.pyplot(fig=plt)
    #Plot 3
    plt.subplot(2,2,3)
    plt.hist(reliance['High'],bins=50, color='green')
    plt.xlabel("High Price")
    plt.ylabel("Frequency")
    plt.title('High')
    st.pyplot(fig=plt)
    #Plot 4
    plt.subplot(2,2,4)
    plt.hist(reliance['Low'],bins=50, color='red')
    plt.xlabel("Low Price")
    plt.ylabel("Frequency")
    plt.title('Low')
    st.pyplot(fig=plt)
    
    sns.heatmap(reliance.corr(),annot=True)
    st.pyplot(fig=plt)


    figure=plt.figure(figsize=(30,10))
    plt.plot(reliance['Volume'])
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title('Date vs Volume')
    st.pyplot(fig=plt)

    plt.figure(figsize=(20,10))
    plt.plot(reliance_ma['Close'],label='Original data')
    plt.plot(reliance_ma['30-day MA'],label='30-MA')
    plt.legend
    plt.title('Stock Price vs 30-day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig=plt)

    plt.figure(figsize=(20,10))
    plt.plot(reliance_ma['Close'],label='Original data')
    plt.plot(reliance_ma['200-day MA'],label='200-MA')
    plt.legend
    plt.title('Stock Price vs 200-day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig=plt)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
    from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
    from sklearn.preprocessing import MinMaxScaler

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM, GRU
    
    from itertools import cycle

    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    print("X_train: ", X_train.shape)
    print("X_test: ", X_test.shape)
    tf.keras.backend.clear_session()
    model=Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=(time_step,1)))
    model.add(LSTM(32,return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    # shift train predictions for plotting

    look_back=time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    print("Train predicted data: ", trainPredictPlot.shape)

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
    print("Test predicted data: ", testPredictPlot.shape)

    names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


    plotdf = pd.DataFrame({'Date': close_stock['Date'],
                       'original_close': close_stock['Close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

    fig = px.line(plotdf,x=plotdf['Date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Stock price','Date': 'Date'})
    fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.pyplot(fig=plt)
    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]

    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat
    last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    new_pred_plot = pd.DataFrame({
        'last_original_days_value':last_original_days_value,
        'next_predicted_days_value':next_predicted_days_value
    })

    names = cycle(['Last 15 days close price','Predicted next 30 days close price'])

    fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                      new_pred_plot['next_predicted_days_value']],
              labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(title_text='Compare last 15 days vs next 30 days',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.pyplot(fig=plt)
    lstmdf=closedf.tolist()
    lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

    names = cycle(['Close price'])

    fig = px.line(lstmdf,labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')

    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.pyplot(fig=plt)


    names = cycle(['SVR', 'RF','KNN','LSTM','GRU'])

    fig = px.line(finaldf[225:], x=finaldf.index[225:], y=[finaldf['svr'][225:],finaldf['rf'][225:], finaldf['knn'][225:], 
                                          finaldf['lstm'][225:], finaldf['gru'][225:],],
             labels={'x': 'Timestamp','value':'Stock close price'})
    fig.update_layout(title_text='Final stock analysis chart', font_size=15, font_color='black',legend_title_text='Algorithms')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.pyplot(fig=plt)






