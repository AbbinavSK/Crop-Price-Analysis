'''App to showcase the conditional volatility forecasting model'''

# Importing the necessary libraries
import streamlit as st

import pandas as pd
import plotly.graph_objects as go
import numpy as np

import arch 
from arch import arch_model
#--------------------------------------------------------------------------------------------------------------------------------------
# Initialising the list of districts as a global variable
mp_districts = ['Ashoknagar', 'Chhindwara', 'Dewas', 'Guna', 'Harda', 'Indore', 'Khandwa', 'Khargone', 'Mandsaur', 
             'Raisen', 'Rajgarh', 'Sagar', 'Sehore', 'Shajapur', 'Shivpuri', 'Tikamgarh', 'Ujjain', 'Vidisha']

# Function to clean the meteorological data: Precipitation and Max Temperature
def cleaning_met_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data.loc['2012-02-01':'2024-10-31']

    for col in data.columns:
        if col not in mp_districts:
            data.drop(col, axis=1, inplace=True)

    data.reset_index(inplace=True)

    return data

# Function to clean and fetch the price log returns
def price_log_returns(data):
    df_logreturns = pd.DataFrame()
    df_logreturns["Price Date"] = data["Price Date"]

    for district in mp_districts:
        df_logreturns[district] = np.log(data[district]) - np.log(data[district].shift(1))
    df_logreturns = df_logreturns.dropna()
    df_logreturns = df_logreturns[:-2]

    return df_logreturns

# Function to clean and fetch the price conditional volatility
def price_cond_vol(df_logreturns):
    df_condvol = pd.DataFrame()
    df_condvol["Price Date"] = df_logreturns["Price Date"]

    for district in mp_districts:
        df_condvol[district] = arch_model(df_logreturns[district], vol='EGARCH', p=1, o=1, q=1).fit(disp='off').conditional_volatility
    df_condvol = df_condvol.dropna()
    
    return df_condvol

# Function to plot a 2D graph in plotly
def plot_graph(x_values, y_values, labels, colors, title, xaxis_title, yaxis_title):
    fig = go.Figure()

    for x, y, label, color in zip(x_values, y_values, labels, colors):
        fig.add_trace(go.Scatter(
            x=x, y=y, 
            mode='lines', 
            name=label,
            line=dict(color=color, width=2)
        ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_dark",
        font=dict(color="white"),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor="black",
        paper_bgcolor="black",
        width=1200,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

#--------------------------------------------------------------------------------------------------------------------------------------
# Title Element:
st.title("Soybean Volatility Forecasting")

# Sidebar Elements:
st.sidebar.header("Parameters")
st.sidebar.write("Adjust the parameters as required")

district = st.sidebar.selectbox('Districts', mp_districts)

#--------------------------------------------------------------------------------------------------------------------------------------
# Importing and cleaning the data
data = pd.read_excel("data\SOYBEAN-MODALPRICE-MONTHLY(Selected).xlsx")
prec_data = pd.read_excel("data\PRECTOTCORR_MONTHLY.xlsx", sheet_name="Madhya Pradesh")
tmax_data = pd.read_excel("data\T2M_MAX_MONTHLY.xlsx", sheet_name="Madhya Pradesh")

data["Price Date"] = pd.to_datetime(data["Price Date"])
prec_data = cleaning_met_data(prec_data)
tmax_data = cleaning_met_data(tmax_data)

df_logreturns = price_log_returns(data)
df_condvol = price_cond_vol(df_logreturns)

# Importing and cleaning the LSTM prediction data
lstm_data = pd.read_csv("Soybean-MP-districtlevel-LSTMpred.csv")
lstm_dates = df_condvol["Price Date"].iloc[-len(lstm_data):].reset_index(drop=True)
df_lstm_pred = pd.DataFrame({
    "Price Date": lstm_dates,
    "LSTM Prediction": lstm_data[district].iloc[:len(lstm_data)].values
})


#--------------------------------------------------------------------------------------------------------------------------------------
# Visualising the Soybean prices
st.markdown("Soybean Modal Price plotted District wise: ")
plot_graph(x_values=[data["Price Date"]], y_values=[data[district]], labels="Modal Price", colors=["cyan"], title=f"Soybean Modal Price in {district}", xaxis_title="Date", yaxis_title="Modal Price (Rs./Quintal)")
# Visualising the Soybean price log returns 
st.markdown("Soybean Price Log Returns plotted District wise: ")
plot_graph(x_values=[df_logreturns["Price Date"]], y_values=[df_logreturns[district]], labels="Log Returns", colors=["green"], title=f"Soybean Price Log Returns in {district}", xaxis_title="Date", yaxis_title="Log Returns")

# Visualising the Soybean price conditional volatility from EGARCH(1, 1, 1) model
st.markdown("Soybean Price Conditional Volatility plotted District wise: ")
plot_graph(x_values=[df_condvol["Price Date"]], y_values=[df_condvol[district]], labels="Conditional Volatility", colors=["magenta"], title=f"Soybean Price Conditional Volatility in {district}", xaxis_title="Date", yaxis_title="Conditional Volatility")

st.markdown("Soybean Price Conditional Volatility with LSTM Prediction:  ")
plot_graph(x_values=[df_condvol["Price Date"], df_lstm_pred["Price Date"]], y_values=[df_condvol[district], df_lstm_pred["LSTM Prediction"]], labels=["Conditional Volatility", "LSTM Prediction"], colors=["red", "cyan"],title=f"Soybean Conditional Volatility vs LSTM Prediction in {district}",xaxis_title="Date",yaxis_title="Conditional Volatility")