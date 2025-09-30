'''App to showcase the analysis done for the paper: The Impact of Meteorological Factors on Crop Price Volatility in India: Case studies of Soybean and Brinjal'''

# Importing the necessary libraries
import streamlit as st

import pandas as pd
import numpy as np
import plotly.graph_objects as go
#--------------------------------------------------------------------------------------------------------------------------------------
# Path to data file and default image
default_image = "India_Map.jpeg"
data_path = "data.csv"

#--------------------------------------------------------------------------------------------------------------------------------------
# Function to read the necessary csv files
def read_csv(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    return df

# Function to plot a 2D graph in plotly
def plot_graph(x_values, y_values, labels, colors, xaxis_title, yaxis_title, y_range=None):
    fig = go.Figure()

    for x, y, label, color in zip(x_values, y_values, labels, colors):
        fig.add_trace(go.Scatter(
            x=x, y=y, 
            mode='lines', 
            name=label,
            line=dict(color=color, width=2)
        ))

    layout_kwargs = dict(
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

    if y_range is not None:
        layout_kwargs["yaxis"] = dict(range=y_range)

    fig.update_layout(**layout_kwargs)
    st.plotly_chart(fig, use_container_width=True)

#--------------------------------------------------------------------------------------------------------------------------------------
# Reading the csv files
data = read_csv(data_path)
states = data["State"].unique()

#--------------------------------------------------------------------------------------------------------------------------------------
# Main body of the Streamlit app
st.title("Crop Price Analysis in India: Case studies of Soybean and Brinjal")
selected_state = st.sidebar.selectbox('Select a State', ['Select a State'] + list(states))

try:
    if selected_state != 'Select a State':
        tabs1, tabs2 = st.tabs(["Temporal", "Spatial"])
        with tabs1:
            state_data = data[data["State"] == selected_state]
            state_data = data[data["State"] == selected_state].sort_values(by="Date")
            crop = state_data["Crop"].unique()[0]

            st.markdown(f"Modal Price of {crop} in {selected_state} from 2012 to 2024: ")
            plot_graph(x_values=[state_data["Date"]], y_values=[state_data["Modal_Price"]], labels="Modal Price", colors=["cyan"], xaxis_title="Date", yaxis_title="Modal Price (INR/Quintal)")

            st.markdown(f"Log Returns of {crop} in {selected_state} from 2012 to 2024: ")
            plot_graph(x_values=[state_data["Date"]], y_values=[state_data["Log_Returns"]], labels="Log Returns", colors=["green"], xaxis_title="Date", yaxis_title="Log Returns", y_range=[-0.81, 0.81])

            st.markdown(f"Squared Log Returns of {crop} in {selected_state} from 2012 to 2024: ")
            plot_graph(x_values=[state_data["Date"]], y_values=[state_data["Squared_Log_Returns"]], labels="Squared Log Returns", colors=["yellow"], xaxis_title="Date", yaxis_title="Squared Log Returns", y_range=[0, 0.63])

            st.markdown(f"Conditional Volatility of {crop} in {selected_state} from 2012 to 2024: ")
            plot_graph(x_values=[state_data["Date"]], y_values=[state_data["Conditional_Volatility"]], labels="Conditional Volatility", colors=["magenta"], xaxis_title="Date", yaxis_title="Conditional Volatility", y_range=[0, 0.7])

            st.markdown(f"SARIMAX and LSTM Predictions of Conditional Volatility for {crop} in {selected_state} during the testing period: ")
            plot_graph(x_values=[state_data["Date"], state_data["Date"], state_data["Date"]], y_values=[state_data["Conditional_Volatility"], state_data["SARIMAX_pred"], state_data["LSTM_pred"]], labels=["Conditional Volatility", "SARIMAX", "LSTM"], colors=["magenta", "orange", "green"], xaxis_title="Date", yaxis_title="Conditional Volatility", y_range=[0, 0.7])
    
        with tabs2:
            volsurface_path_2D = f"{selected_state}_VolatilitySurface_2D.gif"
            volsurface_path_3D = f"{selected_state}_VolatilitySurface_3D.gif"
            st.image(volsurface_path_2D, caption=f"2D Plot for {selected_state}", output_format="GIF")
            st.write("")
            st.image(volsurface_path_3D, caption=f"3D Plot for {selected_state}", output_format="GIF")
    else:
        st.image(default_image, use_container_width=True)

except FileNotFoundError:
    st.error("The required CSV files were not found. Please make sure the required files are in the working directory.")

