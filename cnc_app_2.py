import numpy as np
import streamlit as st
import pickle
from sklearn import linear_model

st.title("CNC Primary Care Service Level Predictor")

def load_model():
    with open('pcp_call_model.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

model = load_model()

def prediction(input1, input2, input3, input4):
    sl = model.predict([[input1, input2, input3, input4]])
    return sl[0]


#inputs - calls offered, AHT, not ready rate, total ftes, fte callouts

st.text("Please fill in the responses below to predict primary care service level")

calls_offered = st.number_input(label="Choose a call volume", min_value=500, max_value=3000, step=1)
aht = st.number_input(label="Average Handle Time (in decimal format, i.e. 5.5 = 5min 30sec -> 0.1 = 6 sec)", min_value=4.0, max_value=7.0, step=0.1)
not_ready = st.number_input(label="Not Ready Rate (%)", min_value=15.0, max_value=40.0, step=0.5)
ftes_logged_in = st.number_input(label="Choose the total number of FTEs logged in for the day (use PowerBI as a guide)", min_value=15.0, max_value=55.0, step=0.5)

not_ready_con = not_ready/100

sl_prediction_temp = prediction(calls_offered, aht, not_ready_con, ftes_logged_in)
sl_prediction = sl_prediction_temp*100

st.header("Service Level Prediction")
if sl_prediction <= 0:
    st.subheader("0")
elif sl_prediction >= 100:
    st.subheader("100")
else:
    st.subheader(sl_prediction)

st.caption("Modeling used is multiple linear regression and was trained on CNC call data Oct 2022 - Jan 2024 (as of Feb 1 2024)")

