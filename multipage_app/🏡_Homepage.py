import streamlit as st
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu

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
st.title('Welcome Page')


st.markdown("""
    Welcome to the Recurrent Neural Network
    This is a simple example of how to create 
    a Recurrent Neural Network using TensorFlow 
    and Keras.
    Please upload your dataset to get started
""")
 


explination_homepage = option_menu("Main Menu", 
                                    ["Recurrent Neural Network",
                                    'Chatbots'], 

                            icons = ['bi-motherboard-fill', 
                                     'bi-robot'], 

                            menu_icon = "cast",

                            orientation = 'horizontal', 


                            default_index = 0)
    
if 'Recurrent Neural Network' in explination_homepage:
    # use of ccs because than we can center the tile otherwise it would be left orientited on the homepage
    st.markdown(f"<div style='text-align:center;'><h1>Recurrent Neural Network</h1></div>",
                unsafe_allow_html=True)
    

    
    st.write("""Picture this: You're on Netflix, craving a good movie night. 
                No need to spend hours scrolling through endless lists, thanks 
                to machine learning! Netflix now tailors recommendations to 
                exactly what you want to see. 🎬 And if hunger strikes mid-movie, 
                Amazon's got your back, suggesting the perfect pizza based on your 
                preferences and order history. 🍕 But here's a fun twist: if you 
                try to identify yourself with a selfie, watch out! The facial 
                recognition program might mistakenly think you're a robot and 
                lock you out. 😄 Don't worry, though – we're working on perfecting 
                that glitch! 🤖✨""")
    
#### Explination of what is Objection Detection
if 'Chatbot' in explination_homepage:

    # use of ccs because than we can center the tile otherwise it would be left orientited on the homepage
    st.markdown(f"<div style='text-align:center;'><h1>Chatbot</h1></div>",
                unsafe_allow_html=True)
    

    
    st.write("""Imagine object recognition as a robot navigating its surroundings, 
                swiftly identifying any object in its path. 🤖 It's akin to 
                having a waiter who, with each new dish served, instantly 
                recognizes its contents, checking for nuts or gluten to alert 
                guests with allergies. 🍽️ Whether it's cars, buildings, or faces, 
                object recognition allows us to identify and track everything in 
                our environment.
                But here's a humorous twist: if you send the object recognition program 
                to a party, it might hilariously attempt to label each pair of shoes as
                a separate object. 👠👞 That might not be the most practical application, 
                but it sure adds a touch of whimsy to the capabilities of object recognition! 😄🌐""")
