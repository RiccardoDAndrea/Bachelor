import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import SimpleRNN, Dropout, Input
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from matplotlib import pyplot as plt
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
    st.dataframe(df.describe())
    st.divider()

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

##################################################################################################
#############  D a t a _ C l e a n i n g #################################################
##################################################################################################



####################################################################################################
#############  D a t a _ V i s u a l i z a t i o n #################################################
####################################################################################################


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

####################################################################################################
#############  D a t a _ V i s u a l i z a t i o n #################################################
####################################################################################################


####################################################################################################
############# R e c c u r e n t _ N e u r a l _ N e t w o r k ######################################
####################################################################################################



with st.expander('Recurrent Neural Network'):
    st.subheader("Create your own Reccurent Neural Network: ")

    forecast_Var = st.selectbox('Enter your Column for the RNN forecast:', 
                                                    options=df.columns, key='RNN Variable')
    y = df[forecast_Var]
    # Abfangen von Fehlern
    # Überprüfung des Datentyps der ausgewählten Variablen
    if y.dtype == 'object' or y.dtype == 'string' or y.dtype == 'datetime64[ns]':
        st.warning('Ups, wrong data type for Target variable!')
        #st_lottie('wrong_data_type_ML.json', width=700, height=300, quality='low', loop=False)
        st.dataframe(df.dtypes, use_container_width=True)
        st.stop()

    if y.dtype == 'object' or y.dtype == 'string' or y.dtype == 'datetime64[ns]':
        st.warning('Ups, wrong data type for Target variable!')
        #st_lottie('wrong_data_type_ML.json', width=700, height=300, quality='low', loop=False)
        st.dataframe(df.dtypes, use_container_width=True)
        st.stop()

    dataset = y.values
    dataset = dataset.astype('float32')  # Konvertieren der Daten in float32
    dataset = np.reshape(dataset, (-1, 1))  # Reshape der Daten in eine 2D-Form

    # Ausgabe der Form des Datensatzes
    Datset_col, Scaled_dataset_col = st.columns(2)
    with Datset_col:
        st.subheader("Dataset Column:" , forecast_Var)

        dtype_col, shape_col = st.columns(2)
        with shape_col:
            st.write("Shape:", str(dataset.shape))
        with dtype_col:
            st.write("Dtype:", str(dataset.dtype))
        st.dataframe(pd.DataFrame(dataset, columns=[forecast_Var]), 
                     use_container_width=True, hide_index=True)
    
    with Scaled_dataset_col:
    # Skalieren der Daten
        st.subheader('Scaled Data')
        scaler = MinMaxScaler(feature_range=(0, 1))  # Auch QuantileTransformer kann ausprobiert werden
        dataset = scaler.fit_transform(dataset)
        
        scaled_dtype_col, scaled_shape_col = st.columns(2)
        with scaled_shape_col:
            st.write("Shape:", str(dataset.shape))
        with scaled_dtype_col:
            st.write("Dtype:", str(dataset.dtype))
        # Ausgabe der skalierten Daten
        
        st.dataframe(pd.DataFrame(dataset, columns=[forecast_Var]), 
                        use_container_width=True, hide_index=True)  # Anzeigen des skalierten Datensatzes in einem DataFrame
    st.info('The dataset has been successfully scaled!')

    total_size = 100  # Total size of the dataset

    # Initial split values
    initial_train_size = 60
    initial_test_size = 40

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

    # Display the sizes and shapes
    # len_dataset_col, train_shape_col, test_shape_col = st.columns(3)
    # with len_dataset_col:
    #     st.markdown(f"**Total dataset size: {len(dataset)}**")
    # with train_shape_col:
    #     st.write(f"**Training set size: {train.shape}**")
    # with test_shape_col:
    #     st.write(f"**Test set size: {test.shape}**")
    
    # Bin size slider for histogram

    train_hist, test_hist = st.columns(2)
    with train_hist:
        train_bin_size = st.slider('Train Bin Size', min_value=1, max_value=100, step=1, value=10, format='%d', key='train_bin_size')
        hist_plot_1 = px.histogram(train, x=train[:, 0], nbins=train_bin_size, labels={'x': 'Feature 1', 'y': 'Count'}, title='Training Set Histogram')
        st.plotly_chart(hist_plot_1)

    # Bin size slider and histogram for test set
    with test_hist:
        test_bin_size = st.slider('Test Bin Size', min_value=1, max_value=100, step=1, value=10, format='%d', key='test_bin_size')
        hist_plot_2 = px.histogram(test, x=test[:, 0], nbins=test_bin_size, labels={'x': 'Feature 1', 'y': 'Count'}, title='Training Set Histogram')
        st.plotly_chart(hist_plot_2)
    

    seq_size_col, seq_size_info_col= st.columns(2)
    with seq_size_col:
        seq_size = st.number_input("Insert a number for the sequence size", min_value=1, max_value=100, value=15, step=1)
    with seq_size_info_col:
        st.info("Sequence size is the number of time steps to look back like a memory of the model.")
    

    trainX, trainY = to_sequences(train, seq_size)
    testX, testY = to_sequences(test, seq_size)
    train_x_col, train_y_col = st.columns(2)
    with train_x_col:
        st.dataframe(trainX, use_container_width=True)
    with train_y_col:
        st.dataframe(trainY, use_container_width=True)
    test_x_col, test_y_col = st.columns(2)
    with test_x_col:
        st.dataframe(testX, use_container_width=True)
    with test_y_col:
        st.dataframe(testY, use_container_width=True)


    st.write("Shape of training set: {}".format(trainX.shape))
    st.write("Shape of test set: {}".format(testX.shape))

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Create the model
    model = Sequential()
    






    ####################################################################################################
    ############# R e c c u r e n t _ N e u r a l _ N e t w o r k ######################################
    ###################################################################################################

