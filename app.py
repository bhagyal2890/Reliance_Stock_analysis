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
