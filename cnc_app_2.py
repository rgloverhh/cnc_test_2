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

calls_offered = st.slider(label="Choose a call volume", min_value=0, max_value=5000)
aht = st.number_input(label="Average Handle Time (in decimal format, i.e. 5 min 30 sec is 5.5)", min_value=4.0, max_value=7.0, step=0.1)
not_ready = st.number_input(label="Not Ready Rate (in decimal format, i.e. 19 percent is .19)", min_value=.15, max_value=.35, step=0.01)
total_fte = st.slider(label="Choose the total number of FTEs staffed on service line", min_value=0, max_value=65)
call_outs = st.slider(label="Choose the estimated FTE call out equivalent", min_value=0, max_value=15)
staffed = total_fte-call_outs

sl_prediction_temp = prediction(calls_offered, aht, not_ready, staffed)
sl_prediction = sl_prediction_temp*100

st.header("Service Level Prediction")
st.subheader(sl_prediction)

