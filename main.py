import streamlit as st
import pandas as pd
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
# Title of the main page
st.title('Recurrent Neural Network')

# Title of the sidebar
# Title for the sidebar

st.sidebar.title('Recurrent Neural Network')
with st.sidebar:
    # File uploader in the sidebar
    file_uploader = st.file_uploader('Upload your dataset', type=['csv'])
    
    # Check if the file has been uploaded
    if file_uploader is None:
        st.info('Please upload your dataset')
        st.stop()
    else:
        # Expander for upload settings
        with st.expander('Upload settings'):
            separator, thousands = st.columns(2)
            with separator:
                selected_separator = st.selectbox('Choose value separator:', (",", ";", ".", ":"))
            with thousands:
                selected_thousands = st.selectbox('Choose thousands separator:', (".", ","), key='thousands')
            
            decimal, __ = st.columns(2)
            with decimal:
                selected_decimal = st.selectbox('Choose decimal separator:', (".", ","), key='decimal')
            with __:
                selected_unicode = st.checkbox('Use Unicode', value=False)

        # Read the uploaded file into a DataFrame with the selected separators
        df = pd.read_csv(file_uploader, sep=selected_separator, 
                         thousands=selected_thousands, decimal=selected_decimal)


    


# Display the DataFrame
st.dataframe(df)
st.divider()

st.markdown("## Data Description")  
st.dataframe(df.describe())



# Remove columns with remove_columns:

st.markdown('Your DataFrame columns')
st.dataframe(df.columns, use_container_width=True)
st.write('Remove your DataFrame columns')

st.subheader("Remove Columns:")
selected_columns = st.multiselect("Choose your columns", df.columns)
df = df.drop(selected_columns, axis=1)
st.dataframe(df)



st.markdown('Your DataFrame data types')
st.dataframe(df.dtypes, use_container_width=True)
st.write('Change your DataFrame data types')

st.subheader("Change your Data Types:")
selected_columns = st.multiselect("Choose your columns", df.columns, key='change_data_type')
selected_dtype = st.selectbox("Choose a data type", ["int64", "float64", "string", "datetime64[ns]"])



options_of_charts = st.multiselect(
                    'What Graphs do you want?', ('Barchart', 
                                                'Linechart', 
                                                'Scatterchart', 
                                                'Histogramm',
                                                'Boxplot'))
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
        fig_scatter = go.Figure(data=scatter_plot_1)
        # Umwandeln des Histogramm-Graphen in eine Bilddatei
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
        st.divider()


st.markdown("Create your own Reccurent Neural Network")
st.write('Prepare your data for training')
Target_variable_col, X_variables_col = st.columns(2)
Target_variable = Target_variable_col.selectbox('Which is your Target Variable (Y)', options=df.columns, key='LR Sklearn Target Variable')
X_variables = X_variables_col.multiselect('Which are your Variables (X)', options=df.columns, key='LR Sklearn X Variables')
# Überprüfung des Datentyps der ausgewählten Variablen
if uploaded_file[Target_variable].dtype == str or uploaded_file[Target_variable].dtype == str :
    st.warning('Ups, wrong data type for Target variable!')
    st_lottie(wrong_data_type_ML, width=700, height=300, quality='low', loop=False)
    st.dataframe(uploaded_file.dtypes, use_container_width=True)
    st.stop()

if any(uploaded_file[x].dtype == object for x in X_variables):
    st.warning('Ups, wrong data type for X variables!')
    st_lottie(wrong_data_type_ML, width=700, height=300, quality='low', loop=False)
    st.dataframe(uploaded_file.dtypes, use_container_width=True)
    st.stop()


if len(X_variables) == 0 :
    st_lottie(no_X_variable_lottie)
    st.warning('X Variable is empty!')
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

    