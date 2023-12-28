import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Setting Webpage Configurations
st.set_page_config(page_icon="⚙",page_title="Rental Price Predictor", layout="wide")

st.title(':green[Smarty] - Your Home Rental Price Predictor 🚀')

@st.cache_resource
def load_model():
    model = pickle.load(open(r'F:\GUVI_DATA_SCIENCE\Project\Smart-Predictive-Modeling-for-Rental-Property-Prices\Artifacts\reg_model.pkl', 'rb'))
    return model

model = load_model()

transformer = pickle.load(open(r'F:\GUVI_DATA_SCIENCE\Project\Smart-Predictive-Modeling-for-Rental-Property-Prices\Artifacts\transformer.pkl','rb'))


df = pd.read_csv(r'F:\GUVI_DATA_SCIENCE\Project\Smart-Predictive-Modeling-for-Rental-Property-Prices\Dataset\analysis_df.csv')

col1,col2,col3 = st.columns(3)

col4,col5,col6 = st.columns(3)

col7,col8,col9 = st.columns(3)

col10,col11,col12 = st.columns(3)

col13,col14,col15 = st.columns(3)

col16,col17,col18 = st.columns(3)

with col1:
    Type = st.selectbox('Select the Type', options = df['type'].value_counts().index.sort_values())

with col2:
    lease_type = st.selectbox('Select the Lease Type', options = df['lease_type'].value_counts().index.sort_values())

with col3:
    gym = st.selectbox('Gym Availability', options = ['Yes', 'No'])

    if gym == 'Yes':
        gym = 1
    else:
        gym = 0

with col4:
    lift = st.selectbox('Lift Availability', options = ['Yes', 'No'] )

    if lift == 'Yes':
        lift = 1
    else:
        lift = 0

with col5:
    swimming_pool = st.selectbox('Swimming pool Availability', options = ['Yes', 'No'] )

    if swimming_pool == 'Yes':
        swimming_pool = 1
    else:
        swimming_pool = 0

with col6:
    negotiable = st.selectbox('Price Negotiable', options = ['Yes', 'No'] )

    if negotiable == 'Yes':
        negotiable = 1
    else:
        negotiable = 0

with col7:
    furnishing = st.selectbox('Furnishing', options = df['furnishing'].value_counts().index.sort_values())


with col8:
    parking= st.selectbox('Parking', options = df['parking'].value_counts().index)

with col9:
    building_type = st.selectbox('Type of Building', options = df['building_type'].value_counts().index)

with col10:
    water_supply = st.selectbox('Water Supply', options = df['water_supply'].value_counts().index)

with col11:
    facing = st.selectbox('Facing Direction', options = df['facing'].value_counts().index)

with col12:
    property_size = st.number_input('Property Size')

log_property_size = np.log(property_size)

with col13:
    property_age = st.number_input('Property Age')

log_property_age = np.log(property_age)

with col14:
    bathroom = st.number_input('Number of Bathrooms')

with col15:
    cup_board = st.number_input('Number of CupBoards')

with col16:
    floor =  st.number_input('Number of Floors')

with col17:
    total_floor = st.number_input('Total Number of Floors')

with col18:
    balconies = st.number_input('Number of Balconies')



user_df = pd.DataFrame([[Type,lease_type,gym,lift,swimming_pool,negotiable,furnishing,parking,log_property_size,log_property_age,bathroom,facing,cup_board,floor,total_floor,water_supply,building_type,balconies]], columns = ['type', 'lease_type', 'gym', 'lift', 'swimming_pool', 'negotiable', 'furnishing', 'parking', 'property_size', 'property_age', 'bathroom', 'facing', 'cup_board', 'floor', 'total_floor', 'water_supply', 'building_type', 'balconies'])

submit = st.button('Predict Home Rental Price')

if submit:
    user_input_transformed = transformer.transform(user_df)
  
    result = model.predict(user_input_transformed)
    st.subheader(f':green[Predicted Home Rental Price] : {result[0]}')