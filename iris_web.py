#import librerias
from os import sep
from click import option
import streamlit as st
import pickle
import pandas as pd 

#extraer archivos pickle
with open('lin_reg.pkl', 'rb')as li:
    lin_reg=pickle.load(li)

with open('log_reg.pkl', 'rb')as lo:
    log_reg=pickle.load(lo)

with open('svc_m.pkl', 'rb')as sv:
    svc_m=pickle.load(sv)

#funcion para clasificar plantas

def classify(num):
    if num==0:
        return 'Setosa'
    elif num==1:
        return 'Versicolor'
    else:
        return 'Virginica'
    

def main():
    st.title('Modelamiento de Iris by Sara')

    st.sidebar.header('User Input Parameters')

    def user_input_parameters():
        sepal_length =st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width =st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length =st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width =st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'sepal_length':sepal_length,
                'Sepal width':sepal_width,
                'Petal length':petal_length,
                'Petal width':petal_width,
                }
        features=pd.DataFrame(data, index=[0])
        return features
    df=user_input_parameters()
    #escoger el modelo
    option = ['Linear Regression', 'Logistic Regression', 'SVM']
    model= st.sidebar.selectbox('Which model you lije to use?', option)

    st.subheader('User Input Parameters')
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        if model == 'Linear Regression':
            st.success(classify(lin_reg.predict(df)))
        elif model == 'Logistic Regression':
            st.success(classify(log_reg.predict(df)))
        else:
            st.success(classify(svc_m.predict(df)))


if __name__ == '__main__':
    main()