import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime
import datetime
import requests
import math
import os 


########################################################################################
#############  L O T T I E _ F I L E S #################################################
########################################################################################
# get the lottie file to display the animation
def load_lottieurl(url:str):
    """ 
    A function to load lottie files from a url

    Input:
    - A URL of the lottie animation
    Output:
    - A lottie animation
    """
    try:
        r = requests.get(url)
        r.raise_for_status()  # Raise an exception for HTTP errors
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading Lottie animation from {url}: {str(e)}")
        return None

no_X_variable_lottie = load_lottieurl('https://assets10.lottiefiles.com/packages/lf20_ydo1amjm.json')
wrong_data_type_ML = load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_2frpohrv.json')
Rnn_welcome_page_lottie = load_lottieurl('https://lottie.host/08c7a53a-a678-4758-9246-7300ca6c3c3f/sLoAgnhaN1.json')
value_is_zero_in_train_size = load_lottieurl('https://assets7.lottiefiles.com/packages/lf20_usmfx6bp.json')

########################################################################################
#############  L O T T I E _ F I L E S #################################################
########################################################################################


########################################################################################
#############  S_I_D_E_B_A_R ###########################################################
########################################################################################

# Set page configuration
st.set_page_config(page_title='exploring-the-power-of-rnns', page_icon=':robot:', layout='wide')
st.title('Recurrent Neural Network')


st.sidebar.title('Recurrent Neural Network')

datasets = ['Upload here your data', 'Weather data for Germany','Yahoo finance API']
selected_dataset = st.sidebar.selectbox('Choose your dataset:', options=datasets)


# Expander for upload settings in the sidebar
with st.sidebar.expander('Upload settings'):

    separator, thousands = st.columns(2)
    
    with separator:
        selected_separator = st.selectbox('Value separator:', (",", ";", ".", ":"))
    
    with thousands:
        selected_thousands = st.selectbox('Thousands separator:', (",", "."), key='thousands')
    
    decimal, unicode = st.columns(2)
    
    with decimal:
        selected_decimal = st.selectbox('Decimal separator:', (".", ","), key='decimal')
    
    with unicode:
        selected_unicode = st.selectbox('File encoding:', ('utf-8', 'utf-16', 'utf-32', 'iso-8859-1', 'cp1252'))

### end of the upload settings expander
# will be uses if the user wants to upload a dataset from the local machine
def load_dataframe(uploaded_file):
    """
    Load a dataframe from the uploaded file with the selected separators.
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, sep=selected_separator, thousands=selected_thousands, decimal=selected_decimal, encoding=selected_unicode)

# gets the data from the yahoo finance API and the weather data for Germany
def load_dataframe_from_url(url):
    """
    Load a dataframe from a URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        content = response.content
        temp_file = 'temp.csv'
        
        with open(temp_file, 'wb') as f:
            f.write(content)
        
        dataset = pd.read_csv(temp_file, sep=selected_separator, thousands=selected_thousands, decimal=selected_decimal, encoding=selected_unicode)
        os.remove(temp_file)
        
        return dataset  # Return DataFrame directly
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading dataset from {url}: {str(e)}")
        return None


# Load the dataset
df = None
if selected_dataset == 'Upload here your data':
    file_uploader = st.sidebar.file_uploader('Upload your dataset', type=['csv'])
    df = load_dataframe(file_uploader)

elif selected_dataset == 'Weather data for Germany':
    dataset_url = "https://raw.githubusercontent.com/RiccardoDAndrea/Bachelor/main/data/processed/Weather_data.csv"
    df = load_dataframe_from_url(dataset_url)



elif selected_dataset == 'Yahoo finance API':

    with st.sidebar.expander('Stock Options'):

        st.info('Please enter the stock you want to analyze and the date range.')
        stock_options = st.text_input("Enter your Stock", value='AAPL')
        stock_options = [stock.strip() for stock in stock_options.split(',')] 
        start_date_col, end_date_col = st.columns(2)

        with start_date_col:
            start_date_input = st.date_input("Start", value=datetime.date(2024, 1, 1))  # Default start date
        
        with end_date_col:
            end_date_input = st.date_input("Last day", value=datetime.date.today())  # Default end date

    data_frames = []
        
    for stock_option in stock_options:

        try:
            data = yf.download(stock_option, start=start_date_input, end=end_date_input)

            if not data.empty:
                data['Stock'] = stock_option  
                data_frames.append(data)

            else:
                st.warning(f"No data found for {stock_option} in the specified date range.")

        except Exception as e:
            st.error(f"Error fetching data for {stock_option}: {str(e)}")
    
    if data_frames:
        df = pd.concat(data_frames)
        df.reset_index(inplace=True)  # Sreset the index
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]  # get only the columns we need



# # Display DataFrame if loaded
# if df is not None:
#     st.subheader("Your DataFrame:")
#     st.dataframe(df)
# else:
#     st.sidebar.info('Please upload your dataset or select a dataset option.')



# Check if the dataframe is loaded

if df is None:
    st.sidebar.info('Please upload your dataset')
    
    st.markdown("""
    ### Create Your Own RNN Architecture 
    """)

    st.info(""" **Start by uploading your own data or using the data stored for you.** 📁""")

    st.write("""        
    1. You can then examine the data and make edits in the **data preprocessing expander**. 🔍🛠️

    2. You also have the option to **visualize the data**. 📊📈

    3. Finally, you can **build your own RNN architecture**. 🧠🔧
    """)
    st.lottie(Rnn_welcome_page_lottie, speed=1, width=800, height=550)
    st.stop()


### General Information about the data
# Display the DataFrame
st.subheader("Your DataFrame: ")
st.dataframe(df, use_container_width=True)
st.divider()

##########################################################################################
#############  D a t a _ d e s c r i b e #################################################
##########################################################################################

with st.expander('Data Description'):
    st.subheader("Data Description: ")  
    st.dataframe(df.describe().round(2), use_container_width=True)

##########################################################################################
#############  D a t a _ d e s c r i b e #################################################
##########################################################################################



##################################################################################################
#############  D a t a _ p r e p r o c e s s i n g _ s t a r t ###################################
##################################################################################################

with st.expander('Data preprocessing'):
    st.subheader('How to proceed with NaN values')
    st.dataframe(df.isna().sum(), use_container_width=True) # get the sum of NaN values in the DataFrame
    checkbox_nan_values = st.checkbox("Do you want to replace the NaN values to proceed?", key="disabled")

    if checkbox_nan_values:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        missing_values = st.selectbox(
            "How do you want to replace the NaN values in the numeric columns?",
            key="visibility",
            options=["None",
                     "Drop rows with NaN values",
                     "interpolate",
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

        elif 'interpolate' in missing_values:
            df = df.interpolate()
            st.write('##### You have succesfully :blue[interpolated the NaN values]')
            st.dataframe(df.isna().sum(), use_container_width=True)

        elif 'Drop rows with NaN values' in missing_values:
            df = df.dropna()
            st.write('##### You have succesfully :blue[drop rows with NaN values]')
            st.dataframe(df.isna().sum(), use_container_width=True)
            
        elif 'with Mean' in missing_values:
            uploaded_file_mean = df[numeric_columns].mean()
            df[numeric_columns] = df[numeric_columns].fillna(uploaded_file_mean)
            st.markdown(' ##### You have succesfully change the NaN values :blue[ with the Mean]')
            st.dataframe(df.isna().sum(), use_container_width=True)

        elif 'with Minimum value' in missing_values:
            uploaded_file_min = df[numeric_columns].min()
            df[numeric_columns] = df[numeric_columns].fillna(uploaded_file_min)
            st.write('##### You have succesfully change the NaN values :blue[with the minimum values]')
            st.dataframe(df.isna().sum(), use_container_width=True)
            
        elif 'with Maximum value' in missing_values:
            uploaded_file_max = df[numeric_columns].max()
            df[numeric_columns] = df[numeric_columns].fillna(uploaded_file_max)
            st.write('##### You have succesfully change the NaN values :blue[with the maximums values]')
            st.dataframe(df.isna().sum(), use_container_width=True)
            
        elif 'with Zero' in missing_values:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            df[numeric_columns] = df[numeric_columns].fillna(0)
            st.write('##### You have successfully changed :blue[the NaN values to 0.]')
            st.dataframe(df.isna().sum(), use_container_width=True)

    st.divider()
    st.subheader("Remove Columns:")
    selected_columns = st.multiselect("Choose your columns", df.columns)
    df = df.drop(selected_columns, axis=1)
    st.dataframe(df, use_container_width=True)
    st.divider()

    st.subheader('Your DataFrame data types: ')
    st.dataframe(df.dtypes, use_container_width=True)
    st.subheader("Change your Data Types:")
    
    change_data_type_col_1, change_data_type_col_2 = st.columns(2)

    with change_data_type_col_1:
        selected_columns_1 = st.multiselect("Choose your columns", df.columns, key='change_data_type_1')
        selected_dtype_1 = st.selectbox("Choose a data type", ['None','int64', 
                                                               'float64', 'string',
                                                               'datime', 'datetime64[ns]'], 
                                                               key='selectbox_1')
        
    with change_data_type_col_2:
        selected_columns_2 = st.multiselect("Choose your columns", df.columns, key='change_data_type_2')
        selected_dtype_2 = st.selectbox("Choose a data type", ['None','int64', 
                                                               'float64', 'string', 'datetime64[ns]'],
                                                               key='selectbox_2')

    # Function to change data types
    def change_data_types(dataframe, columns, dtype):
        if columns:
            try:
                if dtype == "int64":
                    dataframe[columns] = dataframe[columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
                elif dtype == "float64":
                    dataframe[columns] = dataframe[columns].apply(pd.to_numeric, errors='coerce').astype('float64')
                elif dtype == "string":
                    dataframe[columns] = dataframe[columns].astype('string')
                elif dtype == "datetime64[ns]":
                    dataframe[columns] = pd.to_datetime(dataframe[columns], errors='coerce')        

            except Exception as e:
                st.error(f"Error converting columns {columns} to {dtype}: {e}")

    # Apply data type changes
    change_data_types(df, selected_columns_1, selected_dtype_1)
    change_data_types(df, selected_columns_2, selected_dtype_2)

    st.divider()

    # Display the modified DataFrame
    st.subheader('Modified DataFrame data types:')
    st.dataframe(df.dtypes, use_container_width=True)

    # Display the DataFrame
    st.subheader('Modified DataFrame:')
    st.dataframe(df, use_container_width=True)

##################################################################################################
#############  D a t a _ C l e a n i n g #################################################
##################################################################################################



####################################################################################################
#############  D a t a _ V i s u a l i z a t i o n #################################################
####################################################################################################


with st.expander('Data Visualization'):
    st.subheader('Data Visualization')

    options_of_charts = st.multiselect('What Graphs do you want?', 
                                       ('Linechart', 
                                        'Scatterchart',
                                        'Correlation Matrix',
                                        'Histogram'))
    for chart_type in options_of_charts:

        if chart_type == 'Scatterchart':

            st.write('You can freely choose your :blue[Scatter plot]')
            x_axis_val_col_, y_axis_val_col_ = st.columns(2)
            
            with x_axis_val_col_:
                x_axis_val = st.selectbox('Select :blue[X-Axis Value]', options=df.columns, key='x_axis_selectbox')
            
            with y_axis_val_col_:
                y_axis_val = st.selectbox('Select :blue[Y-Axis Value]', options=df.columns, key='y_axis_selectbox')
            scatter_plot_1 = px.scatter(df, x=x_axis_val,y=y_axis_val)

            st.plotly_chart(scatter_plot_1,use_container_width=True)
    
            st.divider()
        
        elif chart_type == 'Linechart':
            st.markdown('You can freely choose your :blue[Linechart] :chart_with_upwards_trend:')

            col3,col4 = st.columns(2)
            
            with col3:
                x_axis_val_line = st.selectbox('Select :blue[X-Axis Value]', options=df.columns,
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

            fig_correlation.update_traces(showscale = False, 
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

        elif chart_type == 'Histogram':
            column_name_col, train_slider_col,  = st.columns(2)
            with column_name_col:
                column_name = st.selectbox('Select column for Histogram', options=df.columns)
            with train_slider_col:
                train_bin_size = st.slider('Train Bin Size', min_value=1, max_value=100, step=1, value=10, format='%d', key='Vis_chart_type')
            hist_plot_1 = px.histogram(df, x=column_name, nbins=train_bin_size, labels={'x': column_name, 'y': 'Count'}, title='Histogram')
            st.plotly_chart(hist_plot_1)

# ####################################################################################################
# #############  D a t a _ V i s u a l i z a t i o n #################################################
# ####################################################################################################


####################################################################################################
############# R e c c u r e n t _ N e u r a l _ N e t w o r k ######################################
####################################################################################################


with st.expander('Recurrent Neural Network'):
    st.subheader("Create your own Reccurent Neural Network: ")

    try:
        forecast_Var = st.selectbox('Enter your Column for the RNN forecast:', 
                                    options=df.columns, key='RNN Variable')

        y = df[[forecast_Var]]
        y = y.dropna()
        
        dataset = y.values
        # dataset = dataset.astype('float32')
        dataset_rounded = np.round(dataset, 2)
        
        st.write("Dataset successfully processed.")
        
    
            

        dataframe_col, hist_col = st.columns(2)

        with dataframe_col:
            st.write(" ")
            st.write(" ")
            y_rounded = pd.DataFrame(dataset_rounded, columns=[forecast_Var])
            st.dataframe(y_rounded, use_container_width=True)

        with hist_col:
            train_bin_size = st.slider('Train Bin Size', min_value=1, max_value=100, step=1, value=10, format='%d', key='train_bin_size')
            hist_plot_1 = px.histogram(y_rounded, x=forecast_Var, nbins=train_bin_size, labels={'x': forecast_Var, 'y': 'Count'}, title='Histogram')
            hist_plot_1.update_layout(width=400, height=360)
            st.plotly_chart(hist_plot_1)

        st.divider()
        #st.write(dataset.shape)
        
        
        # geting the shape of the dataset
        Datset_col, Scaled_dataset_col = st.columns(2)
        with Datset_col:
            st.subheader("Dataset Overview: " , forecast_Var)

            shape_col,dtype_col = st.columns(2)

            with shape_col:
            # Convert shape to string without parentheses
                shape_str = ' , '.join(map(str, dataset.shape))
                st.write("Shape:", shape_str)
            
            with dtype_col:
                st.write(f'The data type: {dataset.dtype}') # für übersichtlichkeit leer gelassen

            
            
            st.dataframe(pd.DataFrame(dataset, columns=[forecast_Var]), 
                        use_container_width=True, hide_index=True)
        
        with Scaled_dataset_col:

            # scaling the data using MinMaxScaler (beacause of the problem of vansihing gradient and exploding gradient)
            st.subheader('Scaled Data Overview:')
            scaler = MinMaxScaler(feature_range=(0, 1))  # Auch QuantileTransformer kann ausprobiert werden
            dataset = scaler.fit_transform(dataset)
            
            # get the shape of the scaled dataset
            scaled_dtype_col, scaled_shape_col = st.columns(2)
            st.dataframe(pd.DataFrame(dataset, columns=[forecast_Var]), 
                                use_container_width=True, hide_index=True)  # Anzeigen des skalierten Datensatzes in einem DataFrame

            with scaled_dtype_col:
                shape_str = ' , '.join(map(str, dataset.shape))
                st.write("Shape:", shape_str)
            
            with scaled_shape_col:
                st.write(f'The data type: {dataset.dtype}')
                
        st.divider()
        total_size = 100            # Total size of the dataset
        initial_train_size = 60     # Initial train size
        initial_test_size = 40      # Initial test size

        # Columns for sliders
        train_size_col, test_size_col = st.columns(2)

        # Synchronize sliders
        with train_size_col:
            train_size = st.slider(
                'Train Size (%)',
                min_value=0,
                max_value=total_size,
                value=initial_train_size,
                key='train_size_slider'
            )

        with test_size_col:
            test_size = st.slider(
                'Test Size (%)',
                min_value=0,
                max_value=total_size,
                value=total_size - train_size,
                key='test_size_slider'
            )

        # Ensure that train_size and test_size sum to total_size
        if train_size + test_size != total_size:
            test_size = total_size - train_size

        # Convert percentage to actual sizes
        train_size_actual = int(len(dataset) * train_size / 100)
        test_size_actual = len(dataset) - train_size_actual

        # Split the dataset
        train, test = dataset[:train_size_actual, :], dataset[train_size_actual:, :]
        
        seq_size_col, seq_size_info_col = st.columns(2)
        
        with seq_size_col:
            st.write(" ")
            seq_size = st.number_input("Insert a number for the sequence size",
                                    min_value=1, max_value=100, 
                                    value=5, step=1)
        
        with seq_size_info_col:
            st.write(" ")
            st.info("Sequence size is the number of time steps to look back like a memory of the model.")
        
        # sequence size like a memory of the model. chunking the data into smaller parts
        def to_sequences(dataset, seq_size=1):
            x = []
            y = []

            for i in range(len(dataset)-seq_size-1):
                #print(i)
                window = dataset[i:(i+seq_size), 0]
                x.append(window)
                y.append(dataset[i+seq_size, 0])
            
            return np.array(x),np.array(y)
        trainX, trainY = to_sequences(train, seq_size)
        testX, testY = to_sequences(test, seq_size)

        
        # Layout for training data
        training_data, test_data = st.tabs(["Training Data", "Test Data"])

        
        with training_data:
            st.write(f"### Training Data X and Y - :blue[{forecast_Var}]")
            st.write("#### Number of time steps to look back")
            train_x_col, train_y_col = st.columns(2)
            with train_x_col:
                st.write("#### Training Data X")
                st.dataframe(trainX, use_container_width=True)
                st.write("Shape of training set: {}".format(trainX.shape))
            
            with train_y_col:
                st.write("#### Training Data Y")
                st.dataframe(trainY, use_container_width=True)
                st.write("Shape of training set: {}".format(trainY.shape))

        with test_data:
            st.write(f"### Test Data X and Y - :blue[{forecast_Var}]")
            # Layout for test data
            test_x_col, test_y_col = st.columns(2)
            
            with test_x_col:
                st.write("#### Test Data X")
                st.dataframe(testX, use_container_width=True)
                st.write("Shape of test set: {}".format(testX.shape))
            
            with test_y_col:
                st.write("#### Test Data Y")
                st.dataframe(testY, use_container_width=True)
                st.write("Shape of test set: {}".format(testY.shape))

        
        
        try:    
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
            
        except IndexError as e:
            if "tuple index out of range" in str(e):
                st.error("An error occurred during reshaping the data. Please check the train or test size and try again.")
                st.info("""
                        Ups something went wrong. Here are some suggestions for improvement:
                        
                        It looks like you have set the train size or the test size to 0 or to a value that is too large.
                        
                        1. **Train Size and Test Size:**
                            - Make sure that the sum of the train size and the test size is equal to 100%.
                            For example:
                            - Train Size: 70%
                            - Test Size: 30%
                            - Total Size: 100%
                        

                        """)
                st.stop()




        # Create the model
        st.divider()
        st.subheader("Create the model Infrastructure:")
        
        # Button to trigger model compilation
        

        # Number of layers input
        number_layers = st.number_input('Number of Layers', min_value=1, max_value=5, 
                                        value=1, step=1)
        #return_sequc = st.checkbox('Return Sequences', value=False) 
        
        st.divider()
        layer_types = []
        units = []
        return_sequences = []
        activations = []

        # UI-Elements for each layer
        for i in range(number_layers):
            st.write(f'Layer {i+1}')
            select_layer_typ_col, select_neurons_col, activation_col, col4 = st.columns(4)
            with select_layer_typ_col:
                layer_type = st.selectbox(f'Layer {i+1} Type', ('Dense', 'LSTM', 'GRU', 'Flatten'), key=f'layer_type_{i}')
                layer_types.append(layer_type)
            with select_neurons_col:
                unit = st.number_input(f'Units in Layer {i+1}', min_value=1, max_value=512, value=64, step=1, key=f'units_{i}')
                units.append(unit)
            with activation_col:
                if layer_type in ['LSTM', 'GRU']:
                    # activation = st.selectbox(f'Activation Function for Dense Layer {i+1}', ('None', 'relu', 'sigmoid', 'tanh', 'softmax'), key=f'activation_{i}')
                    # activations.append(None if activation == 'None' else activation)
                    st.write(" ")
                    st.write(" ")
                    return_seq = st.checkbox(f'Return Sequences in Layer {i+1}', key=f'return_seq_{i}')
                    return_sequences.append(return_seq)
                else:
                    return_sequences.append(None)  # None für Dense Layer
            with col4:
                
                activation = st.selectbox(f'Activation Function for Dense Layer {i+1}', ('None', 'relu', 'sigmoid', 'tanh', 'softmax'), key=f'activation_{i}')
                activations.append(None if activation == 'None' else activation)
                

        # Input for optimizer, loss, epochs, and learning rate
        epochs_col, lr_col = st.columns(2)
        
        with epochs_col:
            epochs = st.number_input('Number of Epochs', 
                                    min_value=1, 
                                    max_value=100, 
                                    value=5, 
                                    step=1)
        with lr_col:
            learning_rate = st.number_input(
                                    "Please insert a learning rate",
                                    min_value=0.0000001,
                                    max_value=0.99999,
                                    value=0.01,
                                    step=0.0001,
                                    format="%.8f"
                        )

        optimizer_col, loss_col = st.columns(2)
        
        with optimizer_col:
            optimizer = st.selectbox('Optimizer', ('adam', 'sgd', 'rmsprop', 'adadelta', 'adagrad'))
        
        with loss_col:
            loss = st.selectbox('Loss', ('mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error'))
        
        # Different layers for the RNN Modell
        
        if st.button('Compile and train the model'):
            model = Sequential()
            for i in range(number_layers):
                if layer_types[i] == 'LSTM':
                    if i == 0:
                        model.add(LSTM(units[i], input_shape=(None, seq_size), return_sequences=return_sequences[i], activation=activations[i]))
                    else:
                        model.add(LSTM(units[i], return_sequences=return_sequences[i], activation=activations[i]))
                elif layer_types[i] == 'GRU':
                    if i == 0:
                        model.add(GRU(units[i], input_shape=(None, seq_size), return_sequences=return_sequences[i], activation=activations[i]))
                    else:
                        model.add(GRU(units[i], return_sequences=return_sequences[i], activation=activations[i]))
                elif layer_types[i] == 'Dense':
                    if i == 0:
                        model.add(Dense(units[i], input_shape=(seq_size,), activation=activations[i]))
                    else:
                        model.add(Dense(units[i], activation=activations[i]))
                elif layer_types[i] == 'Flatten':
                    model.add(Flatten())

            if optimizer == 'adam':
                opt = Adam(learning_rate=learning_rate)
            elif optimizer == 'sgd':
                opt = SGD(learning_rate=learning_rate)
            elif optimizer == 'rmsprop':
                opt = RMSprop(learning_rate=learning_rate)
            elif optimizer == 'adadelta':
                opt = Adadelta(learning_rate=learning_rate)
            elif optimizer == 'adagrad':
                opt = Adagrad(learning_rate=learning_rate)
            else:
                raise ValueError(f'Optimizer "{optimizer}" not recognized.')

            model.compile(loss=loss, optimizer=opt)
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            for line in model_summary:
                st.write(line)
            
            progress_bar = st.progress(0)

            with st.spinner('Model training in progress...'):
                # Placeholder for chat messages
                chat_message_placeholder = st.empty()
                chat_message_placeholder.chat_message("assistant").write("Give me a second, I have to do some math...")

                # Initalias the model results
                train_loss = []
                val_loss = []
                try:
                    for epoch in range(epochs):
                        # Train the model
                        history = model.fit(trainX, trainY, validation_data=(testX, testY), verbose=2, epochs=1)
                        
                        # Speichere den Trainings- und Validierungsverlust
                        train_loss.append(history.history['loss'][0])
                        val_loss.append(history.history['val_loss'][0])

                        # Aktualisiere den Fortschrittsbalken nach jeder Epoche
                        progress_bar.progress((epoch + 1) / epochs)

                    chat_message_placeholder.chat_message('assistant').write('Model training completed!')
                    
                
                    chat_message_placeholder.empty() # delete the chat message

                except Exception as e:
                    #st.write(e)
                    st.error("An error occurred during model training:")
                    st.info("""
                    **Oops, something went wrong. Here are some suggestions for improvement:**

                    1. **Model Architecture:**
                    - Your current architecture may not be optimal. Consider using the following structure as a guide:

                        ```
                        LSTM or GRU, 128 units, return_sequences=True
                        LSTM or GRU, 128 units, return_sequences=False
                        Dense, 1 unit, activation=relu
                        ```

                    - For RNN models, the `return_sequences` parameter is crucial. Set it to `True` if you have more than one layer.

                    2. **Number of Layers:**
                    - The number of layers in your model may be insufficient. For the suggested architecture:

                        ```
                        LSTM or GRU, 128 units, return_sequences=True
                        LSTM or GRU, 128 units, return_sequences=False
                        Dense, 1 unit, activation=relu
                        ```

                    - Consider adding more layers or adjusting the existing ones to enhance performance.
                    """)

                    st.stop()

                
            try:
                    # make predictions
                trainPredict = model.predict(trainX)
                testPredict = model.predict(testX)
                # invert predictions
                
                # st.write(trainPredict.shape, testPredict.shape)
                try:
                    
                    trainPredict = scaler.inverse_transform(trainPredict)
                    trainY = scaler.inverse_transform([trainY])
                    testPredict = scaler.inverse_transform(testPredict)
                    testY = scaler.inverse_transform([testY])
                except ValueError as e:
                    st.error(f"An error occurred during inverse transformation: {e}")
                    st.info("""
                            This could be because you have used `return_sequences=True` in the last layer of an RNN (GRU or LSTM). 
                            Please make sure that `return_sequences=False` is set in the last layer to get an output with the correct dimension. 
                            correct dimension.
                            """)
                    st.stop()

                # st.write(trainPredict.shape, trainY.shape)
                # st.write(testPredict.shape, testY.shape)
                st.write("### Model Evaluation:")


                # Visulasing the training and validation loss
                fig_gradient = go.Figure()
                fig_gradient.add_trace(go.Scatter(x=list(range(epochs)), y=train_loss, mode='lines', name='Training Loss'))
                fig_gradient.add_trace(go.Scatter(x=list(range(epochs)), y=val_loss, mode='lines', name='Validation Loss'))
                fig_gradient.update_layout(
                    title='Training and Validation Loss',
                    xaxis_title='Epochs',
                    yaxis_title='Loss',
                    legend_title='Legend',
                )

                # Display plots
                st.plotly_chart(fig_gradient, key='training_validation_loss')

                

                def format_loss_name(loss_name):
                    return loss_name.replace("_", " ").title()
            
                RMSE_train_com, RMSE_test_com = st.columns(2)
                with RMSE_train_com:
                    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
                    formatted_loss_name = format_loss_name(loss)
                    st.metric(f"Train Score: ({formatted_loss_name})", str(round(trainScore, 2)))


                with RMSE_test_com:
                    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
                    formatted_loss_name = format_loss_name(loss)
                    st.metric(f"Test Score: ({formatted_loss_name})", str(round(testScore, 2)))

                trainPredictPlot = np.empty_like(dataset)
                trainPredictPlot[:, :] = np.nan
                trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict

                # shift test predictions for plotting
                testPredictPlot = np.empty_like(dataset)
                testPredictPlot[:, :] = np.nan
                testPredictPlot[len(trainPredict)+(seq_size*2)+1:len(dataset)-1, :] = testPredict

                dataset_inverse = scaler.inverse_transform(dataset)

                # Check if 'Date' or 'date' is in the DataFrame columns
                if 'Date' in df.columns or 'date' in df.columns:
                    # Create plotly figure
                    fig = go.Figure()

                    # Add traces for the dataset, train prediction, and test prediction
                    fig.add_trace(go.Scatter(
                        x=df['Date'],
                        y=dataset_inverse.flatten(),
                        mode='lines',
                        name='Original Data'
                    ))

                    fig.add_trace(go.Scatter(
                        x=df['Date'][:len(trainPredictPlot)],
                        y=trainPredictPlot.flatten(),
                        mode='lines',
                        name='Train Prediction'
                    ))

                    fig.add_trace(go.Scatter(
                        x=df['Date'][-len(testPredictPlot):],
                        y=testPredictPlot.flatten(),
                        mode='lines',
                        name='Test Prediction'
                    ))

                    # Update layout
                    fig.update_layout(
                        title='Original Data and Predictions',
                        xaxis_title='Date',
                        yaxis_title='Value'
                    )

                    # Display the figure in Streamlit
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    # Create plotly figure
                    fig = go.Figure()

                    # Add traces for the dataset, train prediction, and test prediction
                    fig.add_trace(go.Scatter(
                        x=np.arange(len(dataset_inverse)),
                        y=dataset_inverse.flatten(),
                        mode='lines',
                        name='Original Data'
                    ))

                    fig.add_trace(go.Scatter(
                        x=np.arange(len(trainPredictPlot)),
                        y=trainPredictPlot.flatten(),
                        mode='lines',
                        name='Train Prediction'
                    ))

                    fig.add_trace(go.Scatter(
                        x=np.arange(len(testPredictPlot)),
                        y=testPredictPlot.flatten(),
                        mode='lines',
                        name='Test Prediction'
                    ))

                    # Update layout
                    fig.update_layout(
                        title='Original Data and Predictions',
                        xaxis_title='Time',
                        yaxis_title='Value'
                    )

                    # Display the figure in Streamlit
                    st.plotly_chart(fig, use_container_width=True)

                        
                # Predict the next value
                last_sequence = dataset[-seq_size:]
                last_sequence = np.reshape(last_sequence, (1, 1, seq_size))
                next_value_prediction = model.predict(last_sequence)
                next_value_prediction = scaler.inverse_transform(next_value_prediction)

                # Display the prediction
                st.write("### Next Value Prediction:")
                next_value = round(next_value_prediction[0][0], 2)

                # find the max value of the date so the we can add 1 day to the date
                max_date = df['Date'].max()  # Assuming 'Date' is a column

                # Check the data type of max_date
                if isinstance(max_date, datetime.datetime):
                    
                    next_day = max_date + datetime.timedelta(days=1)
                elif isinstance(max_date, str):
                    
                    try:
                        
                        max_date = datetime.datetime.strptime(max_date, "%Y-%m-%d")
                        next_day = max_date + datetime.timedelta(days=1)

                    except ValueError:
                        # Handle potential format errors (optional)
                        print("Error: Invalid date format. Please check your data.")
                        next_day = None  # Set next_day to None or handle the error as needed
                else:
                    # Handle unexpected data type (optional)
                    print("Error: Unexpected data type for 'Date' column. Please check your data.")
                    next_day = None  # Set next_day to None or handle the error as needed

                # Formatiere das Datum in einen String (optional)
                if next_day is not None:
                    next_day_str = next_day.strftime("%Y-%m-%d")  # Format as needed
                    #st.write(f"### Nächster Tag: {next_day_str}")  
                else:
                    st.write("### Nächster Tag: Berechnung fehlgeschlagen") 
                
                next_day_col, pred_col = st.columns(2)
                with next_day_col:
                    st.metric(label="Next Day", value=next_day_str)
                with pred_col:
                    st.metric(label="Prediction", value=next_value)

            # handleing different type of errors
            except tf.errors.InvalidArgumentError as e:
                error_message = str(e)
                if "Graph execution error: Detected at node" in error_message:
                    st.error("A graph execution error was detected:")
                    st.error(error_message)
                    if "'mean_squared_error/SquaredDifference'" in error_message:
                        st.info("""
                            This could be due to mismatched dimensions between your target values (y_true) and predictions (y_pred). 
                            Please check the shape of your data and ensure that the target values and predictions have compatible dimensions.
                            Ensure that `return_sequences=False` is set in the last layer of an RNN (GRU or LSTM) to get an output 
                            with the correct dimension.
                            """)
                else:
                    st.error(f"A graph execution error occurred: {e}")
                st.stop()

            except ValueError as e:
                if "could not broadcast input array from shape" in str(e):
                    st.error("Oops, an error occurred: could not broadcast input array from shape (14122,2) into shape (14122,1)")
                    st.info("""
                            This issue may be caused by specifying two or more neurons in the output layer of your RNN model instead of just one. Please review your model configuration and ensure that the output layer has only one neuron if this is a requirement.

                            Refer to the following architecture guide for proper configuration:

                                LSTM,   128 units,    return_sequences=True,    (activation = relu)
                                LSTM,   128 units,    return_sequences=False,   (activation = sigmoid)      
                                Dense,  1 unit,       activation=relu
                            
                            or

                                LSTM,   128 units,    return_sequences=True,    (activation = relu)
                                GRU,    128 units,    return_sequences=False,   (activation = tanh)
                                Dense,  1 unit,       activation=relu
                            
                            Ensure that the output layer has exactly one unit to meet the requirements.
                            """)
                    st.stop()
                else:
                    st.error(f"An error occurred: {e}")
                st.stop()

            except Exception as e:
                if "Found array with dim 3. None expected <= 2." in str(e):
                    st.error("An error occurred during inverse transformation: Found array with dim 3. None expected <= 2.")
                    st.info("""
                            This could be because you used `return_sequences=True` in the last layer of an RNN (GRU or LSTM). 
                            Please make sure that `return_sequences=False` is set in the last layer to get an output 
                            with the correct dimension.
                            """)
                else:
                    st.error(f"An error occurred during model evaluation: {e}")
                st.stop()
    
    except ValueError as e:
        error_message = str(e)
        
        if "unsupported operand type(s) for *: 'datetime.date' and 'float'" in error_message:
            st.error(f"""An unexpected error occurred: {e}. It seems like the selected column 
                        contains datetime values that cannot be processed as numerical data.""")
            st.warning("Please make sure that the selected column contains numerical values.")
            st.stop()

        elif 'ValueError: Input 0 of layer "lstm" is incompatible with the layer: expected ndim=3, found ndim=2' in error_message:
            st.error(f"An error occurred during model training: {e}")
            st.info("""
                    This could be because you used `return_sequences=True` in the last layer of an RNN (GRU or LSTM). 
                    Please make sure that `return_sequences=False` is set in the last layer to get an output with the correct dimension.
                    """)
            st.stop()

        else:
            st.error(f"An unexpected error occurred: {e}")
            st.info("""
                        Oops, something went wrong
                        Here is a list of what could be improved :
                        
                        Your architecture does not fit.
                        
                        Use the following structure as a architecture guide :
                            
                            LSTM,   128 units,    return_sequence = True
                            LSTM,   128 units,    return_sequence = False
                            DENSE,  1 unit,      activation = relu 
                    
                            or
                    
                            LSTM,   128 units,    return_sequence = True
                            GRU,    128 units,    return_sequence = False
                            DENSE,  1 unit,      activation = relu 
                        Keep in mind its a RNN Model so `return_sequence` is important. 
                        Set it to True if you have more than 1 layer.                     
                    """)
            st.stop()

    except TypeError as e:
        if  "can't multiply sequence by non-int of type 'float'" in str(e):
            st.error(f"An error occurred: {e}")
            st.info(f"""
                    The column used for the RNN forecast, **'{forecast_Var}'**, contains non-numeric values and cannot be processed as numerical data. The current data type of this column is **{df[forecast_Var].dtypes}**. Please ensure that all values are numeric.

                    - In the **Data Preprocessing** section (accessible via the expander), you have the option to change the data type of this column.
                    - The column should contain numeric values (either integers or floats) for the model to be trained effectively.

                    Please verify and correct the data type to ensure proper training of the model.
                    """)

                                
           
            
    except NameError as e:
        st.error(f"A NameError occurred: {e}")
        st.error("""This may be due to a variable or function name being used before it's defined. Please check your code and correct any naming issues.""")
        st.stop()

    except Exception as e:
        st.write(e)
        st.error(f"An unexpected error occurred: {e}")
        st.error("""(2). This is an unknown error for us. Please report it to us so that we 
                    can investigate and fix it. Contact: riccardo.dandrea@hs-osnabrück.de""")
        st.stop()
    

####################################################################################################
############# R e c c u r e n t _ N e u r a l _ N e t w o r k ######################################
###################################################################################################
