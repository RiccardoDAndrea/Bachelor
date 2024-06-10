import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import requests
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from matplotlib import pyplot as plt

####<--- Lottie Files --->####
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

####<--- Lottie Files --->####



# Title of the main page
st.set_page_config(page_title='Recurrent Neural Network', page_icon=':robot:', layout='wide')
st.title('Recurrent Neural Network')


st.sidebar.title('Recurrent Neural Network')
file_uploader = st.sidebar.file_uploader('Upload your dataset', type=['csv'])
    
# Check if the file has been uploaded
if file_uploader is None:           # If no file is uploaded
    st.sidebar.info('Please upload your dataset')
    st.markdown("""
        Welcome to the Recurrent Neural Network
        This is a simple example of how to create 
        a Recurrent Neural Network using TensorFlow 
        and Keras.
        Please upload your dataset to get started
        """)
    
    st_lottie(no_data_lottie)
    st.stop()       # Stop the script so that we dont get an error

else:
    # Expander for upload settings.
    with st.sidebar.expander('Upload settings'):
        separator, thousands = st.columns(2)
        with separator:
            selected_separator = st.selectbox('value separator:', (",", ";", ".", ":"))
        with thousands:
            selected_thousands = st.selectbox('thousands separator:', (".", ","), key='thousands')
        
        decimal, unicode = st.columns(2)
        with decimal:
            selected_decimal = st.selectbox('decimal separator:', (".", ","), key='decimal')
        with unicode:
            selected_unicode = st.selectbox('file encoding:', ('utf-8', 'utf-16', 'utf-32', 'iso-8859-1', 'cp1252'))

   

        # Read the uploaded file into a DataFrame with the selected separators
df = pd.read_csv(file_uploader, sep=selected_separator, 
                thousands=selected_thousands, decimal=selected_decimal)


    

### General Information about the data
# Display the DataFrame
st.subheader("Your DataFrame: ")
st.dataframe(df, use_container_width=True)
st.divider()

with st.expander('Data Description'):
    st.subheader("Data Description: ")  
    st.dataframe(df.describe())
    st.divider()

### General Information about the data end


### Data Cleaning
with st.expander('Data Cleaning'):
    st.subheader('How to proceed with NaN values')
    st.dataframe(df.isna().sum(), use_container_width=True) # get the sum of NaN values in the DataFrame
    checkbox_nan_values = st.checkbox("Do you want to replace the NaN values to proceed?", key="disabled")

    if checkbox_nan_values:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        missing_values = st.selectbox(
            "How do you want to replace the NaN values in the numeric columns?",
            key="visibility",
            options=["None",
                    "with Median", 
                    "with Mean", 
                    "with Minimum value", 
                    "with Maximum value", 
                    "with Zero"])

        if 'with Median' in missing_values:
            uploaded_file_median = df[numeric_columns].median()
            df[numeric_columns] = df[numeric_columns].fillna(uploaded_file_median)
            st.write('##### You have succesfully change the NaN values :blue[with the Median]')
            st.dataframe(df.isna().sum(), use_container_width=True)
            st.divider()
            
        elif 'with Mean' in missing_values:
            uploaded_file_mean = df[numeric_columns].mean()
            df[numeric_columns] = df[numeric_columns].fillna(uploaded_file_mean)
            st.markdown(' ##### You have succesfully change the NaN values :blue[ with the Mean]')
            st.dataframe(df.isna().sum(), use_container_width=True)
            st.divider()

        elif 'with Minimum value' in missing_values:
            uploaded_file_min = df[numeric_columns].min()
            df[numeric_columns] = df[numeric_columns].fillna(uploaded_file_min)
            st.write('##### You have succesfully change the NaN values :blue[with the minimum values]')
            st.dataframe(df.isna().sum(), use_container_width=True)
            st.divider()
            
        elif 'with Maximum value' in missing_values:
            uploaded_file_max = df[numeric_columns].max()
            df[numeric_columns] = df[numeric_columns].fillna(uploaded_file_max)
            st.write('##### You have succesfully change the NaN values :blue[with the maximums values]')
            st.dataframe(df.isna().sum(), use_container_width=True)
            st.divider()
            
        elif 'with Zero' in missing_values:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            df[numeric_columns] = df[numeric_columns].fillna(0)
            st.write('##### You have successfully changed :blue[the NaN values to 0.]')
            st.dataframe(df.isna().sum(), use_container_width=True)
            st.divider()

    st.divider()
    st.subheader("Remove Columns:")
    selected_columns = st.multiselect("Choose your columns", df.columns)
    df = df.drop(selected_columns, axis=1)
    st.dataframe(df)
    st.divider()


    st.subheader('Your DataFrame data types: ')
    st.dataframe(df.dtypes, use_container_width=True)
    st.write('Change your DataFrame data types')

    st.subheader("Change your Data Types:")
    selected_columns = st.multiselect("Choose your columns", df.columns, key='change_data_type')
    selected_dtype = st.selectbox("Choose a data type", ["int64", "float64", "string", "datetime64[ns]"])
    st.divider()

    ### Data Cleaning end 

### Data Visualization

with st.expander('Data Visualization'):

    options_of_charts = st.multiselect('What Graphs do you want?', ('Linechart', 
                                                                    'Scatterchart',
                                                                    'Correlation Matrix'))
    for chart_type in options_of_charts:

        if chart_type == 'Scatterchart':
            st.write('You can freely choose your :blue[Scatter plot]')
            x_axis_val_col_, y_axis_val_col_ = st.columns(2)
            with x_axis_val_col_:
                x_axis_val = st.selectbox('Select X-Axis Value', options=df.columns, key='x_axis_selectbox')
            with y_axis_val_col_:
                y_axis_val = st.selectbox('Select Y-Axis Value', options=df.columns, key='y_axis_selectbox')
            scatter_plot_1 = px.scatter(df, x=x_axis_val,y=y_axis_val)

            st.plotly_chart(scatter_plot_1,use_container_width=True)
            # Erstellen des Histogramms mit Plotly
            plt.tight_layout()
            st.divider()
        
        elif chart_type == 'Linechart':
            st.markdown('You can freely choose your :blue[Linechart] :chart_with_upwards_trend:')

            col3,col4 = st.columns(2)
            
            with col3:
                x_axis_val_line = st.selectbox('Select X-Axis Value', options=df.columns,
                                            key='x_axis_line_multiselect')
            with col4:
                y_axis_vals_line = st.multiselect('Select :blue[Y-Axis Values]', options=df.columns,
                                                key='y_axis_line_multiselect')

            line_plot_1 = px.line(df, x=x_axis_val_line, y=y_axis_vals_line)
            st.plotly_chart(line_plot_1)
        
        elif chart_type == 'Correlation Matrix':
            corr_matrix = df.select_dtypes(include=['float64', 
                                                    'int64']).corr()


            # Erstellung der Heatmap mit Plotly
            fig_correlation = px.imshow(corr_matrix.values, 
                                        color_continuous_scale = 'purples', 
                                        zmin = -1, 
                                        zmax = 1,
                                        x = corr_matrix.columns, 
                                        y = corr_matrix.index,
                                        labels = dict( x = "Columns", 
                                                    y = "Columns", 
                                                    color = "Correlation"))

            # Anpassung der Plot-Parameter
            fig_correlation.update_layout(
                                        title='Correlation Matrix',
                                        font=dict(
                                        color='grey'
                )
            )

            fig_correlation.update_traces(  showscale = False, 
                                            colorbar_thickness = 25)

            # Hinzufügen der numerischen Werte als Text
            annotations = []
            for i, row in enumerate(corr_matrix.values):
                for j, val in enumerate(row):
                    annotations.append(dict(x=j, y=i, text=str(round(val, 2)), showarrow=False, font=dict(size=16)))
            fig_correlation.update_layout(annotations=annotations)

            # Anzeigen der Plot
            st.plotly_chart(fig_correlation, use_container_width= True)
            fig_correlationplot = go.Figure(data=fig_correlation)








st.subheader("Create your own Reccurent Neural Network: ")

Target_variable_col, X_variables_col = st.columns(2)

Target_variable = Target_variable_col.selectbox('Which is your Target Variable (Y)', 
                                                options=df.columns, key='RNN Variable')

X_variables = X_variables_col.multiselect('Which are your Variables (X)', 
                                          options=df.columns, key='RNN X Variables')

# Überprüfung des Datentyps der ausgewählten Variablen
if df[Target_variable].dtype == str or df[Target_variable].dtype == str :
    st.warning('Ups, wrong data type for Target variable!')
    st_lottie(wrong_data_type_ML, width=700, height=300, quality='low', loop=False)
    st.dataframe(df.dtypes, use_container_width=True)
    st.stop()


if any(df[x].dtype == object for x in X_variables):
    st.warning('Ups, wrong data type for X variables!')
    st_lottie(wrong_data_type_ML, width=700, height=300, quality='low', loop=False)
    st.dataframe(df.dtypes, use_container_width=True)
    st.stop()


if len(X_variables) == 0 :
    st_lottie(no_X_variable_lottie)
    st.warning('X Variable is empty!')
    st.stop()

total_size = 100
train_size = 60
test_size = 40

train_size_col, test_size_col = st.columns(2)

with train_size_col:
    train_size = st.slider('Train Size', min_value=0, max_value=total_size, value=train_size, key= 'Sklearn train size')
    test_size = total_size - train_size

with test_size_col:
    test_size = st.slider('Test Size', min_value=0, max_value=total_size, value=test_size, key= 'Sklearn test size')
    train_size = total_size - test_size

# Relevant damit das Skript weiter läuft und nicht immer in Fehlermeldungen läuft
if train_size <= 0:
    st_lottie(value_is_zero_in_train_size, width=700, height=300, quality='low', loop=False)
    st.warning('Train size should be greater than zero.')
    st.stop()

elif test_size <= 0:
    st.warning('Test size should be greater than zero.')
    st_lottie(value_is_zero_in_train_size, width=700, height=300, quality='low', loop=False)
    st.stop()

elif train_size + test_size > len(df):
    st.warning('Train size and Test size exceed the number of samples in the dataset.')
    st_lottie(value_is_zero_in_train_size, width=700, height=300, quality='low', loop=False)
    st.stop()
    
elif train_size == len(df):
    st.warning('Train size cannot be equal to the number of samples in the dataset.')
    st_lottie(value_is_zero_in_train_size, width=700, height=300, quality='low', loop=False)
    st.stop()




# # Prepare the data for training
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

# # Create a Sequential model
# model = Sequential()
# model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# # Compile the model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train the model
# model.fit(X, y, epochs=150, batch_size=10)

# # Evaluate the model
# _, accuracy = model.evaluate(X, y)

# # Display the accuracy
# st.write('Accuracy: %.2f' % (accuracy*100))

    