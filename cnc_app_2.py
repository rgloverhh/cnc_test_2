import numpy as np
import streamlit as st
import pickle
from sklearn import linear_model

st.title("CNC Primary Care Service Level Predictor")

def load_model():
    with open('pcp_call_model.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model



#inputs - calls offered, AHT, not ready rate, total ftes, fte callouts

st.text("Please fill in the responses below to predict primary care service level")

model = load_model()

calls_offered = st.slider(label="Choose a call volume", min_value=0, max_value=5000)
aht = st.number_input(label="Average Handle Time (in decimal format, i.e. 5:30 is 5.5)", min_value=0, max_value=10, format="%.2f")
not_ready = st.number_input(label="Not Ready Rate (in decimal format, i.e. 19 percent is .19)", min_value=0, max_value=1, format="%.2f")
total_fte = st.slider(label="Choose the total number of FTEs staffed on service line", min_value=0, max_value=65)
call_outs = st.slider(label="Choose the estimated FTE call out equivalent", min_value=0, max_value=15)

feature_list = [[calls_offered, aht, not_ready, total_fte-call_outs]]

sl_prediction = model.predict(feature_list)

st.title(f"Predicted Service Level: {int(sl_prediction)}")

