#import librerias
from codecs import strict_errors
from os import sep
from click import option
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression;
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score;

import matplotlib.pyplot as plt;
import streamlit as st
import pickle
import pandas as pd 
import os
import numpy as np;


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
    st.title('Proyecto 2')
    st.subheader("Sara Paulina Medrano Cojulún   201908053") 

    uploaded_files = st.file_uploader("Cargue el Archivo de Entrada",  accept_multiple_files=True)
    print(uploaded_files)
    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        

    algoritmos = st.selectbox("Algoritmos: ", 
                     ['Seleccione una opción','Regresión lineal', 'Regresión polinomial', 'Clasificador Gaussiano', 'Clasificador de árboles de decisión', 'Redes neuronales']) 
    ope = st.selectbox("Operaciones : ", 
                     ['Seleccione una opción','Graficar puntos', 'Definir función de tendencia (lineal o polinomial)', 'Realizar predicción de la tendencia (según unidad de tiempo ingresada)', 'Clasificar por Gauss o árboles de decisión o redes neuronales']) 
    if(ope == 'Realizar predicción de la tendencia (según unidad de tiempo ingresada)'):
        unidadPre= st.text_input("Ingrese la unidad de tiempo", "")

    if(algoritmos == 'Regresión polinomial'):
         grado = st.text_input("Ingrese el grado de la función Polinomial", "")
    
    Paramx = st.text_input("Ingrese el nombre del parametro X", "")
    Paramy = st.text_input("Ingrese el nombre del parametro Y", "")
    
    


    if(st.button('Generar')): 

    #verificacion de archivo valido

        #result = name.title()
        print(name)
        root, extension = os.path.splitext(name)
        print('Root:', root)
        print('extension:', extension)  
        if(extension.upper() == ".CSV"):
            df = pd.read_csv(name)
        elif(extension.upper() == ".json"):
            df = pd.read_json(name)
        elif(extension.upper() == ".xlsx"):
            df = pd.read_excel(name)
        else:
            st.error("Tipo de Archivo no valido")

        #print(df)
        st.subheader("RESULTADOS")

        #Algoritmos

        #Algoritmo de regresion Lineal

        if(algoritmos == 'Regresión lineal'):
            VarX = np.asarray(df[Paramx]).reshape(-1, 1)
            VarY = df[Paramy]
            print("estoy en lineal")
            linear_regression = LinearRegression()
            linear_regression.fit(VarX, VarY)
            Y_pred = linear_regression.predict(VarX)
            r2=r2_score(VarY, Y_pred)
            if(ope == 'Graficar puntos'):
                fig = plt.figure(figsize=(12,9))
                plt.scatter(VarX, VarY, color='green')
                plt.plot(VarX, Y_pred, color='blue')

                st.subheader("Grafica de Puntos")
                st.pyplot(fig)
            elif(ope == 'Realizar predicción de la tendencia (según unidad de tiempo ingresada)'):
                st.subheader("Predicción de Tendencia")
                st.success(linear_regression.predict([[int(unidadPre)]]))
            elif(ope == 'Definir función de tendencia (lineal o polinomial)'):
                st.subheader("Función de tendencia")
                st.success("y = " + str(linear_regression.coef_[0]) + "x + " + str(linear_regression.intercept_))
                st.success("Coeficiente: " + str(linear_regression.coef_[0]) )
                st.success("Intercepción: "+ str(linear_regression.intercept_))
                st.success("r^2: "+ str(r2))
                st.success('Error cuadrático:'+  str(mean_squared_error(VarY, Y_pred)))
             
            elif(ope == 'Clasificar por Gauss o árboles de decisión o redes neuronales'):
                st.error("Operación no validad para este Algoritmo")
            else:
                st.error("No selecciono ninguna Operación") 

        #Algoritmo de regresion Polinomial
        elif(algoritmos == 'Regresión polinomial'):
            VarX = np.asarray(df[Paramx]).reshape(-1, 1)
            VarY = df[Paramy]
            polinomial = PolynomialFeatures(degree = int(grado))
            X_trans= polinomial.fit_transform(VarX)
            linear_regression = LinearRegression()
            linear_regression.fit(X_trans,VarY)
            Y_pred=linear_regression.predict(X_trans)
            if(ope == 'Graficar puntos'):
                fig = plt.figure(figsize=(12,9))
                plt.scatter(VarX, VarY, color='green')
                plt.plot(VarX, Y_pred, color='red')
                st.subheader("Grafica de Puntos")
                st.pyplot(fig)
            elif(ope == 'Realizar predicción de la tendencia (según unidad de tiempo ingresada)'):
                st.subheader("Predicción de Tendencia")
                st.success(linear_regression.predict(polinomial.fit_transform([[int(unidadPre)]])))
            elif(ope == 'Definir función de tendencia (lineal o polinomial)'):

                polinimio= 'Y ='
                gra=int(grado)
                for poli in linear_regression.coef_:
                    polinimio = polinimio +  '+' + str(poli) + 'X^' + str(gra) 
                    gra=gra-1
                #print(polinimio)               
                st.subheader("Función Polinomial")
                st.success(polinimio)
                st.success("Error Cuadratico Medio: " + str(mean_squared_error(VarY, Y_pred, squared=False)))
                st.success("Raíz del Error Cuadratico Medio: "+ str(np.sqrt(mean_squared_error(VarY, Y_pred, squared=False))))
                st.success("Coeficiente de Determinacion R2: " + str(r2_score(VarY, Y_pred)))
                

            elif(ope == 'Clasificar por Gauss o árboles de decisión o redes neuronales'):
                st.error("Operación no validad para este Algoritmo")
            else:
                st.error("No selecciono ninguna Operación") 


        #Algoritmo de Clasificador Gaussiano
        elif(algoritmos== 'Clasificador Gaussiano'):
            print('Clasificador Gaussiano')
        elif(algoritmos == 'Clasificador de árboles de decisión'):
            print('Clasificador de árboles de decisión')
        elif(algoritmos == 'Redes neuronales'):
            print('Redes neuronales')
        else:
                st.error("No selecciono ningun algoritmo") 
    
     

         
if __name__ == '__main__':
    main()