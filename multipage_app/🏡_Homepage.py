import streamlit as st
from streamlit_lottie import st_lottie
import requests
import math
import os 



########################################################################################
#############  L O T T I E _ F I L E S #################################################
########################################################################################
def load_lottieurl(url:str): 
    """ 
    A funcztion to load lottie files from a url

    Input:
    - A URL of the lottie animation
    Output:
    - A lottie animation
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

no_X_variable_lottie = load_lottieurl('https://assets10.lottiefiles.com/packages/lf20_ydo1amjm.json')
wrong_data_type_ML = load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_2frpohrv.json')
no_data_lottie = load_lottieurl('https://lottie.host/08c7a53a-a678-4758-9246-7300ca6c3c3f/sLoAgnhaN1.json')
value_is_zero_in_train_size = load_lottieurl('https://assets7.lottiefiles.com/packages/lf20_usmfx6bp.json')

########################################################################################
#############  L O T T I E _ F I L E S #################################################
########################################################################################

# Title of the main page
st.set_page_config(page_title='exploring-the-power-of-rnns', page_icon=':robot:', layout='wide')
st.title('Recurrent Neural Network')


st.markdown("""
    Welcome to the Recurrent Neural Network
    This is a simple example of how to create 
    a Recurrent Neural Network using TensorFlow 
    and Keras.
    Please upload your dataset to get started
""")
 




    

