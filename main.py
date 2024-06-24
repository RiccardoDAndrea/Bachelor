import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import requests
import math
import os 
from sklearn.preprocessing import MinMaxScaler

# function
def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset)-seq_size-1):
        #print(i)
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])
        
    return np.array(x),np.array(y)

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


st.sidebar.title('Recurrent Neural Network')
file_uploader = st.sidebar.file_uploader('Upload your dataset', type=['csv'])


def dataframe():
    """
    The following function gives the User the capability to 
    enter a dataframe that he wants
    """
    uploaded_file = st.sidebar.file_uploader('Upload here your file', key='dataframe')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=st.session_state.separator)
        return df
        
    

def dataframe_from_url(url):
    response = requests.get(url)
    content = response.content

    # Speichern des Inhalts als temporäre Datei
    temp_file = 'temp.csv'
    with open(temp_file, 'wb') as f:
        f.write(content)

    # Laden der CSV-Datei mit Pandas
    dataset = pd.read_csv(temp_file, sep= st.session_state.separator)
    os.remove(temp_file)
    dataset_regression = pd.DataFrame(dataset)
    return dataset_regression
  
datasets = ['German DAX Data', 'APPLE stock Data', 'Own dataset']  # Liste der verfügbaren Datensätze
selected_datasets = st.sidebar.selectbox('Choose your Dataset:', options=datasets)

if 'German DAX Data' in selected_datasets:
    dataset_url = "https://raw.githubusercontent.com/RiccardoDAndrea/Bachelor/main/data/raw/DAX_Data.csv"
    uploaded_file = dataframe_from_url(dataset_url)

elif 'APPLE stock Data' in selected_datasets:
    dataset_url = ""
    uploaded_file = dataframe_from_url(dataset_url)

elif 'Own dataset' in selected_datasets:
    uploaded_file = dataframe()   
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

# Spalte 'Date' in datetime-Objekte konvertieren
# Sicherstellen, dass die 'Date'-Spalte im DateTime-Format ist
df['Date'] = pd.to_datetime(df['Date'])

# # Extrahieren Sie das Datum ohne die Zeitkomponente
df['Date'] = df['Date'].dt.date




    

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
    st.dataframe(df.describe(), use_container_width=True)

##########################################################################################
#############  D a t a _ d e s c r i b e #################################################
##########################################################################################



##################################################################################################
#############  D a t a _ C l e a n i n g _ e n d #################################################
##################################################################################################

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

    # Column 1: Select columns and data type
    with change_data_type_col_1:
        selected_columns_1 = st.multiselect("Choose your columns", df.columns, key='change_data_type_1')
        selected_dtype_1 = st.selectbox("Choose a data type", ["None","int64", "float64", "string", "datetime64[ns]"], key='selectbox_1')

    # Column 2: Select columns and data type
    with change_data_type_col_2:
        selected_columns_2 = st.multiselect("Choose your columns", df.columns, key='change_data_type_2')
        selected_dtype_2 = st.selectbox("Choose a data type", ["None", "int64", "float64", "string", "datetime64[ns]"], key='selectbox_2')

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
                    dataframe[columns] = dataframe[columns].apply(pd.to_datetime, errors='coerce')
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
                                        'Correlation Matrix'))
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
        
    except TypeError as e:
        if "unsupported operand type(s) for *: 'datetime.date' and 'float'" in str(e):
            st.error(f"An unexpected error occurred: {e}. It seems like the selected column contains datetime values that cannot be processed as numerical data.")
            st.warning("Please make sure that the selected column contains numerical values.")
            st.stop()
        else:
            st.error(f"An unexpected error occurred: {e}")
            st.error("""This is still an unknown error for us. Report it to us so that we 
                          can fix it at the following address: riccardo.dandrea@hs-osnabrück.de""")
            
            st.stop()
            

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
    
    
    # Ausgabe der Form des Datensatzes
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
    # Skalieren der Daten
        st.subheader('Scaled Data Overview:')
        scaler = MinMaxScaler(feature_range=(0, 1))  # Auch QuantileTransformer kann ausprobiert werden
        dataset = scaler.fit_transform(dataset)
        
        scaled_dtype_col, scaled_shape_col = st.columns(2)
        st.dataframe(pd.DataFrame(dataset, columns=[forecast_Var]), 
                            use_container_width=True, hide_index=True)  # Anzeigen des skalierten Datensatzes in einem DataFrame
    
        with scaled_dtype_col:
            shape_str = ' , '.join(map(str, dataset.shape))
            st.write("Shape:", shape_str)
        
        with scaled_shape_col:
            st.write(f'The data type: {dataset.dtype}')
            
    st.divider()
    total_size = 100        # Total size of the dataset
    initial_train_size = 60 # Initial train size
    initial_test_size = 40  # Initial test size

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
    
    trainX, trainY = to_sequences(train, seq_size)
    testX, testY = to_sequences(test, seq_size)

    
    
    
    # Layout for training data
    training_data, test_data = st.tabs(["Training Data", "Test Data"])

    
    
    with training_data:
        st.write(f"### Forecast Training Data X and Y - :blue[{forecast_Var}]")
        train_x_col, train_y_col = st.columns(2)
        with train_x_col:
            st.write("#### Number of time steps to look back")
            st.dataframe(trainX, use_container_width=True)
            st.write("Shape of training set: {}".format(trainX.shape))
        
        with train_y_col:
            st.write("#### Forecast Training Data Y")
            st.dataframe(trainY, use_container_width=True)
            st.write("Shape of training set: {}".format(trainY.shape))

    with test_data:
        st.write(f"### Forecast Test Data X and Y - :blue[{forecast_Var}]")
        # Layout for test data
        test_x_col, test_y_col = st.columns(2)
        
        with test_x_col:
            st.write("#### Forecast Test Data X")
            st.dataframe(testX, use_container_width=True)
            st.write("Shape of test set: {}".format(testX.shape))
        
        with test_y_col:
            st.write("#### Forecast Test Data Y")
            st.dataframe(testY, use_container_width=True)
            st.write("Shape of test set: {}".format(testY.shape))

    
    

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



    # Create the model
    st.divider()
    st.subheader("Create the model Infrastructure:")
    
    # Button to trigger model compilation
    

    # Number of layers input
    number_layers = st.number_input('Number of Layers', min_value=1, max_value=4, value=1, step=1)
    #return_sequc = st.checkbox('Return Sequences', value=False) 
    

    st.divider()
    layer_types = []
    units = []
    return_sequences = []
    activations = []

    # UI-Elemente zur Eingabe der Layer-Konfiguration
    for i in range(number_layers):
        st.write(f'Layer {i+1}')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            layer_type = st.selectbox(f'Layer {i+1} Type', ('Dense', 'LSTM', 'GRU', 'Flatten'), key=f'layer_type_{i}')
            layer_types.append(layer_type)
        with col2:
            unit = st.number_input(f'Units in Layer {i+1}', min_value=1, max_value=512, value=64, step=1, key=f'units_{i}')
            units.append(unit)
        with col3:
            if layer_type in ['LSTM', 'GRU']:
                st.write(" ")
                st.write(" ")
                return_seq = st.checkbox(f'Return Sequences in Layer {i+1}', key=f'return_seq_{i}')
                return_sequences.append(return_seq)
            else:
                return_sequences.append(None)  # None für Dense Layer
        with col4:
            if layer_type == 'Dense':
                activation = st.selectbox(f'Activation Function for Dense Layer {i+1}', ('None', 'relu', 'sigmoid', 'tanh', 'softmax'), key=f'activation_{i}')
                activations.append(None if activation == 'None' else activation)
            else:
                activations.append(None)

    # Eingabe für Epochen, Optimizer und Loss-Funktion
    epochs_col, lr_col = st.columns(2)
    with epochs_col:
        epochs = st.number_input('Number of Epochs', min_value=1, max_value=100, value=5, step=1)
    with lr_col:
        learning_rate = st.number_input('Learning Rate', min_value=0.0000, max_value=0.1, value=0.001, step=0.0001)

    optimizer_col, loss_col = st.columns(2)
    with optimizer_col:
        optimizer = st.selectbox('Optimizer', ('adam', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam', 'ftrl'))
    with loss_col:
        loss = st.selectbox('Loss', ('mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'cosine_similarity', 'huber', 'logcosh', 'poisson', 'kullback_leibler_divergence', 'kl_divergence'))

    if st.button('Compile and train the model'):
        model = Sequential()
        for i in range(number_layers):
            if layer_types[i] == 'LSTM':
                if i == 0:
                    model.add(LSTM(units[i], input_shape=(None, seq_size), return_sequences=return_sequences[i]))
                else:
                    model.add(LSTM(units[i], return_sequences=return_sequences[i]))
            elif layer_types[i] == 'GRU':
                if i == 0:
                    model.add(GRU(units[i], input_shape=(None, seq_size), return_sequences=return_sequences[i]))
                else:
                    model.add(GRU(units[i], return_sequences=return_sequences[i]))
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
        elif optimizer == 'adamax':
            opt = Adamax(learning_rate=learning_rate)
        elif optimizer == 'nadam':
            opt = Nadam(learning_rate=learning_rate)
        elif optimizer == 'ftrl':
            opt = Ftrl(learning_rate=learning_rate)
        else:
            raise ValueError(f'Optimizer "{optimizer}" not recognized.')

        model.compile(loss=loss, optimizer=opt)
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        for line in model_summary:
            st.write(line)

        # Trainingscode hier einfügen
        model.fit(trainX, trainY, validation_data=(testX, testY), 
                verbose=2, epochs=epochs)
        
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        # Überprüfen der Form der Ausgabe
        
        # st.write(trainPredict.shape, testPredict.shape)
        try:
            
            trainPredict = scaler.inverse_transform(trainPredict)
            trainY = scaler.inverse_transform([trainY])
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform([testY])
        except ValueError as e:
            st.error(f"An error occurred during inverse transformation: {e}")
            st.stop()


        # st.write(trainPredict.shape, trainY.shape)
        # st.write(testPredict.shape, testY.shape)
        st.write("### Model Evaluation:")

        RMSE_train_com, RMSE_test_com = st.columns(2)
        with RMSE_train_com:
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
            st.metric('Train Score: RMSE',round(trainScore,2))

        with RMSE_test_com:
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
            st.metric('Test Score: RMSE', round(testScore,2))

        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(seq_size*2)+1:len(dataset)-1, :] = testPredict

        dataset_inverse = scaler.inverse_transform(dataset)

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
        st.plotly_chart(fig)
        
        # Predict the next value
        last_sequence = dataset[-seq_size:]
        last_sequence = np.reshape(last_sequence, (1, 1, seq_size))
        next_value_prediction = model.predict(last_sequence)
        next_value_prediction = scaler.inverse_transform(next_value_prediction)

        # Display the prediction
        st.write("### Next Value Prediction:")
        next_value_df = pd.DataFrame({
            'Date': [pd.Timestamp.now().normalize() + pd.Timedelta(days=1)],
            'Predicted Value': [next_value_prediction[0][0]]
        })
        st.dataframe(next_value_df)
        
        
    
        


        
        
        


    
    
    





####################################################################################################
############# R e c c u r e n t _ N e u r a l _ N e t w o r k ######################################
###################################################################################################
