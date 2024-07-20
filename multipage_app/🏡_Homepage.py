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

welcome_greeting_lottie = load_lottieurl('https://lottie.host/8cbd3d9c-ee94-43e8-af85-1a8c2351c44a/4Ac9zxkEQi.json')
Rnn_explanation_lottie = load_lottieurl('https://lottie.host/08c7a53a-a678-4758-9246-7300ca6c3c3f/sLoAgnhaN1.json')
deep_learning_explanation_lottie = load_lottieurl('https://lottie.host/2deebe4e-3eac-4557-9d7f-e2ae62e362d3/IFbhMAF4e8.json')
chatbot_explanation_lottie = load_lottieurl('https://lottie.host/855882cd-791e-4a74-b538-798f742ae262/CpCWasudWO.json')
########################################################################################
#############  L O T T I E _ F I L E S #################################################
########################################################################################

# Title of the main page
st.set_page_config(page_title='exploring-the-power-of-rnns', page_icon=':robot:', layout='wide')
st.sidebar.title('Your Journey Starts Here 🚀')
st.title(' Welcomen by Exploring the Power of RNNs and LLMs 🧠🤖📈🌐✨')


st.markdown("""

    Ready to dive into the exciting world of Recurrent Neural Networks (RNNs)? 🤖
    This simple example will guide you through creating an RNN using TensorFlow 
    and Keras. Whether you're a seasoned data scientist 🧑‍🔬 or just curious about 
    AI, there's something here for everyone. 🌟

    :blue[Start your journey by exploring the explanations *below*]. 📚 Discover what sets 
    Deep Learning apart from Recurrent Neural Networks, and find out what these 
    chatty chatbots are all about! 💬

    Oh, and see that :blue[little blue arrow at the top right in the corner]
    of your screen? Click on it to get started by uploading your own data 📂 or using 
    our demo data. Let's embark on this AI adventure together! 🚀

""")

st_lottie(welcome_greeting_lottie, width=1200, height=400, key="initial")
 


explination_homepage = option_menu("Main Menu", 
                                    ["Deep Learning",
                                     "Recurrent Neural Network",
                                     "Chatbots"], 

                            icons = ["bi-cpu-fill",
                                     "bi-motherboard-fill", 
                                     "bi-robot"], 

                            menu_icon = "cast",

                            orientation = 'horizontal', 


                            default_index = 0)


if 'Deep Learning' in explination_homepage:
    st.markdown(f"<div style='text-align:center;'><h1>Deep Learning</h1></div>",
                unsafe_allow_html=True)
    
    st.write("""Deep learning is a subset of machine learning where artificial 
                neural networks, inspired by the human brain, learn from large 
                amounts of data. 🧠 These networks consist of layers of nodes, 
                each layer processing information and passing it on to the next, 
                much like neurons in our brains. Deep learning excels in tasks 
                like image and speech recognition, natural language processing, 
                and even game playing. 🕵️‍♂️ Imagine training a dog: at first, 
                it might not understand your commands, but with consistent training, 
                it learns to respond correctly. Similarly, deep learning algorithms 
                improve their performance as they are fed more data and undergo more 
                training cycles, uncovering patterns and making accurate predictions 
                or decisions. 📊🔍""")
    st.lottie(deep_learning_explanation_lottie, speed=1, 
              width=1200, height=400, key="Deep_Learning",
              quality="low")


if 'Recurrent Neural Network' in explination_homepage:
    st.markdown(f"<div style='text-align:center;'><h1>Recurrent Neural Network</h1></div>",
                unsafe_allow_html=True)
    
    st.write("""Recurrent Neural Networks (RNNs) are a type of neural network 
                designed to recognize patterns in sequences of data, such as text, 
                time series, or even video frames. 🔄 Unlike traditional neural 
                networks, RNNs have loops allowing information to persist, making 
                them ideal for tasks where context and sequence matter. 📝 
                Think of RNNs as the memory keeper of neural networks. Imagine 
                you're reading a story – you need to remember what happened in 
                the previous sentences to understand the current one. Similarly, 
                RNNs maintain a memory of previous inputs to influence the current 
                output, making them powerful for language modeling, translation, 
                and even predicting stock prices. 📈🧩""")
    st.lottie(Rnn_explanation_lottie, speed=1, width=1200, 
              height=400, key="RNN",
              quality="low")

    
#### Explination of what is Objection Detection
if 'Chatbot' in explination_homepage:
    st.markdown(f"<div style='text-align:center;'><h1>Chatbot</h1></div>",
                unsafe_allow_html=True)
    
    st.write("""Chatbots are AI-driven programs designed to simulate human 
                conversation. They can interact with users via text or voice, 
                providing instant responses and assistance. 💬 Imagine having 
                a personal assistant available 24/7, capable of answering your 
                questions, helping with tasks, or even making small talk. Chatbots 
                use natural language processing (NLP) to understand and generate 
                human language, making interactions feel natural and engaging. 
                🤖 Whether you're asking about the weather, troubleshooting a 
                technical issue, or booking a flight, chatbots can streamline 
                these interactions, saving time and providing efficient service. 
                🌐✨ And as technology advances, chatbots are becoming even more 
                sophisticated, able to handle complex queries and offer more 
                personalized experiences. 🚀""")
    
    st_lottie(chatbot_explanation_lottie, width=1200, height=400, 
              key="chatbot",
              quality="low")
    

