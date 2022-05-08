# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 19:49:01 2022

@author: adity
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly_express as px
from PIL import Image

selected = option_menu(
    menu_title = "Navigation Bar",
    options = ["Home","Data Visualization","Model Prediction"],
    icons = ["house","graph-up","clipboard-data"],
    menu_icon = "cast",
    default_index = 0,
    orientation = "horizontal"
    )

if selected == "Model Prediction":
## Model Prediction ##    
    st.write("""
    # Breast Cancer Prediction App
    
    This app predicts the type of **Breast Cancer** cell.         
            
    """)
    
    st.sidebar.header("User Input Features")
    
    st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)                    
    """)
    
    ## Collect user input features into dataframe
    
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file",type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_feature():
            texture_mean = st.sidebar.slider("Texture Mean",9.71,39.28,18.84)
            concave_points_mean = st.sidebar.slider("Concave Points Mean",0.000,0.201,0.1)
            area_se = st.sidebar.slider("Area Standard Error",6.802,542.2,100.00)
            texture_worst = st.sidebar.slider("Texture Worst",12.02,49.54,25.648)
            perimeter_worst = st.sidebar.slider("Perimeter Worst",50.41,251.2,100.00)
            area_worst = st.sidebar.slider("Area Worst",185.2,4254.0,849.906)
            #smoothness_worst = st.sidebar.slider("Smoothness Worst",0.071,0.222,0.1)
            #concavity_worst = st.sidebar.slider("Concavity Worst",0.000,1.252,0.5)
            concave_points_worst = st.sidebar.slider("Concave Points Worst",0.000,0.291,0.115)
            #symmetry_worst = st.sidebar.slider("Symmetry Worst",0.1565,0.6638,0.45)
            
            # data = {"texture_mean": texture_mean,
            #         "concave_points_mean": concave_points_mean,
            #         "area_se": area_se,
            #         "texture_worst": texture_worst,
            #         "perimeter_worst": perimeter_worst,
            #         "area_worst": area_worst,
            #         "smoothness_worst": smoothness_worst,
            #         "concavity_worst": concavity_worst,
            #         "concave_points_worst": concave_points_worst,
            #         "symmetry_worst": symmetry_worst
            #         }
            
            data = {"texture_mean": texture_mean,
                    "concave_points_mean": concave_points_mean,
                    "area_se": area_se,
                    "texture_worst": texture_worst,
                    "perimeter_worst": perimeter_worst,
                    "area_worst": area_worst,
                    "concave_points_worst": concave_points_worst,
                    }
            
            features = pd.DataFrame(data,index=[0])
            return features
        input_df = user_input_feature()
    length = len(input_df)
        
    # Combine user_input with entire penguin dataset
    
    cancer_raw = pd.read_csv("data.csv")
    cancer_raw["concave_points_mean"] = cancer_raw.iloc[:,-4]
    cancer_raw["concave_points_worst"] = cancer_raw.iloc[:,9]
    #cancer = cancer_raw[["texture_mean","concave_points_mean","area_se","texture_worst","perimeter_worst","area_worst","smoothness_worst","concavity_worst","concave_points_worst","symmetry_worst"]]
    cancer = cancer_raw[["texture_mean","concave_points_mean","area_se","texture_worst","perimeter_worst","area_worst","concave_points_worst"]]
    df = pd.concat([input_df,cancer],axis=0)  
    
    # Printing the User Inputs as the Output
    st.subheader("User Input Features")
    
    if uploaded_file is not None:
        st.write(df[:length])
    else:
        st.write("Awaiting CSV to be uploaded. Currently using example input parameters (shown below).")
        st.write(df[:length])
    
    # Scaling the data Using Standard Scaler
    
    #from sklearn.preprocessing import StandardScaler
    
    #scaler = StandardScaler()
    #df = scaler.fit_transform(df)
    # =============================================================================
    # mean = [0.048552,34.959487,25.648453,849.907821,0.114606]
    # sd = [0.048552,24.294515,6.054406,475.645240,0.065732]
    # df[:length] = (df[:length]-mean)/sd
    # =============================================================================
    df = df[:length]
         
    # Read in pickled model
    
    load_clf = pickle.load(open("cancer.pkl","rb"))
    
    # Apply model to make predictions
    
    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)
    
    st.subheader("Prediction")
    cancer_type = np.array(["Benign","Malignant"])
    st.write(cancer_type[prediction])
    
    st.subheader("Prediction Probability")
    st.write(prediction_proba)
    
if selected == "Home":
    #st.subheader("Breast Cancer")
    st.sidebar.subheader("Sections")
    with st.sidebar:
        home_options = option_menu(
            menu_title = None,
            options = ["Breast Cancer","Informative Sources","About the project"],
            default_index = 0)
    if home_options == "Breast Cancer":
        with st.container():
            st.subheader("About Breast Cancer")
            st.write("---")
            col1,col2 = st.columns(2)
            ## To add image on the image column
            #image = Image.open('Breast_Cancer.jpg')
            
            col1.image("https://www.cdc.gov/cancer/breast/basic_info/images/female-breast-diagram-750px.jpg?_=41558", caption='Breast Anatomical View')
            ## To add text in the text column
            col2.markdown("""
            Breast cancer is a disease in which cells in the breast grow out of control.There are different kinds of breast cancer. The kind of breast cancer depends on which cells in the breast turn into cancer.
            Breast cancer can begin in different parts of the breast. A breast is made up of three main parts:
            * Lobules 
            * Ducts
            * Connective tissue
            The lobules are the glands that produce milk. The ducts are tubes that carry milk to the nipple. The connective tissue (which consists of fibrous and fatty tissue) surrounds and holds everything together. Most breast cancers begin in the ducts or lobules.
            Breast cancer can spread outside the breast through blood vessels and lymph vessels. When breast cancer spreads to other parts of the body, it is said to have metastasized.
            Further info can be explored from this [link](https://www.cdc.gov/cancer/breast/basic_info/what-is-breast-cancer.htm).
         """)
    if home_options == "About the project":
        st.markdown("""
                    This website was developed as part of the 4th Semester Thesis titled "**A COMPARATIVE STUDY OF MACHINE LEARNING ALGORITHMS FOR BREAST CANCER DETECTION AND PREDICTIVE TOOL BUILDING USING PYTHON**".
                    """)
                    
    if home_options == "Informative Sources":
        st.subheader("Understanding Breast Cancer")
        st.video("https://www.youtube.com/watch?v=KyeiZJrWrys&t=6s")
        st.subheader("The Role of Family History in Breast Cancer")
        st.video("https://www.youtube.com/watch?v=RkG4L50CbvA")
        st.subheader("Risk Reduction and Warning Signs")
        st.video("https://www.youtube.com/watch?v=SwoJQ-Ui7pY")
        st.subheader("Breast Cancer Treatment Plans")
        st.video("https://www.youtube.com/watch?v=6ML_bv2SVPw")
        st.markdown("""
        ### Additional Resources:
        * [Centers for Disease Control and Prevention](https://www.cdc.gov/cancer/breast/basic_info/what-is-breast-cancer.htm)
        * [National Breast Cancer Foundation, Inc](https://www.nationalbreastcancer.org/breast-cancer-treatment/)
        * [WHO](https://www.who.int/news-room/fact-sheets/detail/breast-cancer?msclkid=07e79184ceaa11ec8f38032f8f405c46)
                    """)
        
# =============================================================================
#     image_column, text_column = st.columns((1,2))
#     with image_column:
#             # To insert image to the image column
#     with text_column:
#             #st.subheader("")
#         st.write(
#                 )
# =============================================================================
    
if selected == "Data Visualization":
    st.title("Data Visualization")
    
    st.sidebar.subheader("Visualization Settings")
    
    # Inputing datafile
    uploaded_file = st.sidebar.file_uploader(label="Upload a csv or xlsx file else by default will use Breast Cancer Winsconsin (Diagnostics) data.",
                             type = ['csv','xlsx'])
    
    #global data
    if uploaded_file is not None:
        print(uploaded_file)
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            data = pd.read_excel(uploaded_file)
    else:
        data = pd.read_csv('data.csv')
        data = data.drop(['id','Unnamed: 32'],axis=1)
    
    ## Showing the Dataframe of the data being used
    st.dataframe(data)
    
    numerical_columns = list(data.select_dtypes(['float','int']).columns)
    char_column = list(data.select_dtypes('object').columns)
    char_column.append(None)
    
    ## To create a chart selection widget in the sidebar
    chart_selected=st.sidebar.selectbox(label = "Select the type of Visualization.", options=['Scatterplot','Histogram','Boxplot','Heatmap'])
    
    
    if chart_selected == 'Scatterplot':
        st.sidebar.subheader("Scatterplot Settings")
        x_values = st.sidebar.selectbox('X axis',options= numerical_columns)
        y_values = st.sidebar.selectbox('Y axis',options= numerical_columns)
        z_value = st.sidebar.selectbox("Categorical Value",options = char_column)
        plot = px.scatter(data_frame=data,x = x_values,y = y_values,color = z_value)
        st.plotly_chart(plot)
    elif chart_selected == 'Histogram':
        st.sidebar.header("Histogram Settings")
        x_values = st.sidebar.selectbox('X axis',options= numerical_columns)
        z_value = st.sidebar.selectbox("Categorical Value",options = char_column)
        nbins = st.sidebar.slider("Number of Bins",min_value = 5, max_value = 50)
        plot = px.histogram(data_frame = data,x = x_values,nbins=nbins,color=z_value)
        st.plotly_chart(plot)
    elif chart_selected == 'Boxplot':
        st.sidebar.header("Boxplot Settings")
        y_values = st.sidebar.selectbox('Y axis',options= numerical_columns)
        z_value = st.sidebar.selectbox("Categorical Value",options = char_column)
        plot = px.box(data_frame = data,y=y_values,color = z_value)
        st.plotly_chart(plot)
    elif chart_selected == 'Heatmap':
        corr = data.corr() # Generate correlation matrix
        mask = np.triu(np.ones_like(corr, dtype=np.bool))
        corr = corr.mask(mask)
        plot = px.imshow(corr)
        st.plotly_chart(plot)
        
        