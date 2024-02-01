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

calls_offered = st.slider(label="Choose a call volume", min_value=500, max_value=4000)
aht = st.number_input(label="Average Handle Time (in decimal format, i.e. 5 min 30 sec is 5.5)", min_value=4.0, max_value=7.0, step=0.1)
not_ready = st.number_input(label="Not Ready Rate (%)", min_value=15, max_value=40, step=1)
total_fte = st.number_input(label="Choose the total number of FTEs staffed on service line", min_value=15, max_value=55, step=1)
call_outs = st.number_input(label="Choose the estimated FTE call out equivalent", min_value=0, max_value=15)
staffed = total_fte-call_outs
not_ready_con = not_ready/100

sl_prediction_temp = prediction(calls_offered, aht, not_ready_con, staffed)
sl_prediction = sl_prediction_temp*100

st.header("Service Level Prediction")
if sl_prediction <= 0:
    st.subheader("0")
elif sl_prediction >= 100:
    st.subheader("100")
else:
    st.subheader(sl_prediction)

st.caption("Modeling used is multiple linear regression and was trained on CNC call data Oct 2022 - Jan 2024*")
st.caption("*as of 2/1/24")

