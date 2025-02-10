'''App to showcase the Crop Price Analysis: Madhya Pradesh'''

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
def plot_graph(x_values, y_values, labels, colors, xaxis_title, yaxis_title):
    fig = go.Figure()

    for x, y, label, color in zip(x_values, y_values, labels, colors):
        fig.add_trace(go.Scatter(
            x=x, y=y, 
            mode='lines', 
            name=label,
            line=dict(color=color, width=2)
        ))

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_dark",
        font=dict(color="white"),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor="black",
        paper_bgcolor="black",
        width=1200,
        height=400,
    
        legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
    )
    st.plotly_chart(fig, use_container_width=True)

#--------------------------------------------------------------------------------------------------------------------------------------
# Title Element:
st.title(body="Crop Price Analysis: Soybean in Madhya Pradesh", anchor="center")

# Sidebar Elements:
st.sidebar.header("Input Parameters")
analysis_type = st.sidebar.radio("Select Analysis Level", ["District-Level", "State-Level"])

#--------------------------------------------------------------------------------------------------------------------------------------
# Importing and cleaning the data
data = pd.read_excel("SOYBEAN-MODALPRICE-MONTHLY(Selected).xlsx")
data["Price Date"] = pd.to_datetime(data["Price Date"])

df_logreturns = price_log_returns(data)
df_condvol = price_cond_vol(df_logreturns)

#--------------------------------------------------------------------------------------------------------------------------------------
# State-Level Analysis
if analysis_type == "State-Level":

    st.markdown("**Changes in Soybean Price Volatility across Madhya Pradesh:**")
    st.image("Volatility-Surface-MP(2020-24)(1).gif", use_container_width=True)
    st.image("Volatility-Surface(2020-24)(1).gif", use_container_width=True)

# District-Level Analysis
elif analysis_type == "District-Level":
    district = st.sidebar.selectbox('Districts', mp_districts)
    lstm_data = pd.read_csv("Soybean-MP-districtlevel-LSTMpred.csv")
    lstm_dates = df_condvol["Price Date"].iloc[-len(lstm_data):].reset_index(drop=True)
    df_lstm_pred = pd.DataFrame({
        "Price Date": lstm_dates,
        "LSTM Prediction": lstm_data[district].iloc[:len(lstm_data)].values
    })
    # Visualising the Soybean prices
    st.markdown(f"**Soybean Modal Price plotted for {district}:**")
    plot_graph(x_values=[data["Price Date"]], y_values=[data[district]], labels="Modal Price", colors=["cyan"], xaxis_title="Date", yaxis_title="Modal Price (Rs./Quintal)")

    # Visualising the Soybean price log returns 
    st.markdown(f"**Soybean Price Log Returns plotted for {district}:**")
    plot_graph(x_values=[df_logreturns["Price Date"]], y_values=[df_logreturns[district]], labels="Log Returns", colors=["green"], xaxis_title="Date", yaxis_title="Log Returns")

    # Visualising the Soybean price conditional volatility from EGARCH(1, 1, 1) model
    st.markdown(f"**Soybean Price Conditional Volatility for {district}:**")
    plot_graph(x_values=[df_condvol["Price Date"]], y_values=[df_condvol[district]], labels="Conditional Volatility", colors=["magenta"], xaxis_title="Date", yaxis_title="Conditional Volatility")

    # Visualising the Soybean price conditional volatility with LSTM prediction
    st.markdown(f"**Soybean Price Conditional Volatility with LSTM Prediction for {district}:**")
    plot_graph(x_values=[df_condvol["Price Date"], df_lstm_pred["Price Date"]], y_values=[df_condvol[district], df_lstm_pred["LSTM Prediction"]], labels=["Conditional Volatility", "LSTM Prediction"], colors=["red", "cyan"],xaxis_title="Date",yaxis_title="Conditional Volatility")
