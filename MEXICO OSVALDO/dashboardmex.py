
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from funpymodeling.exploratory import freq_tbl 
import matplotlib.pyplot as plt
import scipy.special as special
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


@st.cache_resource

def load_data():
    df1=pd.read_csv("Mexico sin outliers.csv")
    df2=pd.read_csv("Mexico sin outliers.csv", index_col= 'host_name')
    df3=pd.read_csv("Mexico logistic.csv")
    
    #Extraccion de caracteristicas. Analisis univariado de las variables categoricas mas significativas
    #host_is_superhost
    table= freq_tbl(df1['host_is_superhost'])
    Filtro= table[table['frequency']>1]
    Filtro_index1= Filtro.set_index('host_is_superhost')  

    #Selecciono columnas numericas de df3
    numeric_df9 = df3.select_dtypes(['float','int'])  
    numeric_cols9= numeric_df9.columns   

    #Selecciono columnas numericas de df2
    numeric_df8 = df2.select_dtypes(['float','int'])  
    numeric_cols8= numeric_df8.columns   

    #Selecciono las columnas tipo numericas del dataframe Filtro_index1
    numeric_df1 = Filtro_index1.select_dtypes(['float','int']) 
    numeric_cols1= numeric_df1.columns  

    #host_response_time
    table2= freq_tbl(df1['host_response_time'])
    Filtro2= table2[table2['frequency']>1]
    Filtro_index2= Filtro2.set_index('host_response_time')  

    #Selecciono las columnas tipo numericas del dataframe Filtro_index2
    numeric_df2 = Filtro_index2.select_dtypes(['float','int']) 
    numeric_cols2= numeric_df2.columns

    #host_identity_verified
    table3= freq_tbl(df1['host_identity_verified'])
    Filtro3= table3[table3['frequency']>1]
    Filtro_index3= Filtro3.set_index('host_identity_verified')  

    #Selecciono las columnas tipo numericas del dataframe Filtro_index3
    numeric_df3 = Filtro_index3.select_dtypes(['float','int']) 
    numeric_cols3= numeric_df3.columns

    #room_type
    table4= freq_tbl(df1['room_type'])
    Filtro4= table4[table4['frequency']>1]
    Filtro_index4= Filtro4.set_index('room_type')  

    #Selecciono las columnas tipo numericas del dataframe Filtro_index4
    numeric_df4 = Filtro_index4.select_dtypes(['float','int']) 
    numeric_cols4= numeric_df4.columns

    #property_type
    table5= freq_tbl(df1['property_type'])
    Filtro5= table5[table5['frequency']>1]
    Filtro_index5= Filtro5.set_index('property_type')  

    #Selecciono las columnas tipo numericas del dataframe Filtro_index5
    numeric_df5 = Filtro_index5.select_dtypes(['float','int']) 
    numeric_cols5= numeric_df5.columns

    #neighbourhood_cleansed
    table6= freq_tbl(df1['neighbourhood_cleansed'])
    Filtro6= table6[table6['frequency']>1]
    Filtro_index6= Filtro6.set_index('neighbourhood_cleansed')  

    #Selecciono las columnas tipo numericas del dataframe Filtro_index6
    numeric_df6 = Filtro_index6.select_dtypes(['float','int']) 
    numeric_cols6= numeric_df6.columns

    #instant_bookable
    table7= freq_tbl(df1['instant_bookable'])
    Filtro7= table7[table7['frequency']>1]
    Filtro_index7= Filtro7.set_index('instant_bookable')  

    #Selecciono las columnas tipo numericas del dataframe Filtro_index7
    numeric_df7 = Filtro_index7.select_dtypes(['float','int']) 
    numeric_cols7= numeric_df7
    
    return Filtro_index1, Filtro_index2, Filtro_index3, Filtro_index4, Filtro_index5, Filtro_index6, Filtro_index7, df1, df2, df3, numeric_df1, numeric_df2, numeric_df3, numeric_df4, numeric_df5, numeric_df6, numeric_df7, numeric_df8, numeric_df9, numeric_cols1, numeric_cols2, numeric_cols3, numeric_cols4, numeric_cols5, numeric_cols6, numeric_cols7, numeric_cols8, numeric_cols9

#Cargo los datos obtenidos de la función "load_data"
Filtro_index1, Filtro_index2, Filtro_index3, Filtro_index4, Filtro_index5, Filtro_index6, Filtro_index7, df1, df2, df3, numeric_df1, numeric_df2, numeric_df3, numeric_df4, numeric_df5, numeric_df6, numeric_df7, numeric_df8, numeric_df9, numeric_cols1, numeric_cols2, numeric_cols3, numeric_cols4, numeric_cols5, numeric_cols6, numeric_cols7, numeric_cols8, numeric_cols9 = load_data()

#CREACION DE DASHBOARD
#SIDEBAR
st.sidebar.title("DASHBOARD MEXICO")
st.sidebar.header("Sidebar")
st.sidebar.subheader("Panel de selección")

#FRAMES
Frames= st.selectbox(label= "Analisis de correlaciones", options= ["Regresion lineal simple", "Regresion lineal multiple", "Regresion no lineal", "Regresion logistica", "Otras graficas"])

#Regresion lineal simple
#Declaramos las variables dependientes e independientes para la regresión lineal
if Frames == "Regresion lineal simple":
    st.title("REGRESION LINEAL SIMPLE")
    st.header("Selecciona las variables")
    check_box = st.sidebar.checkbox(label= "Mostrar Dataset")
    if check_box:
        st.write(Filtro_index1)
        st.write(df2)
    x_selected = st.sidebar.selectbox(label = "x (Variable independiente)", options= numeric_cols8)
    y_selected = st.sidebar.selectbox(label = "y (Variable dependiente)", options= numeric_cols8)
    figure1= px.scatter(data_frame=numeric_df8, x=x_selected, y= y_selected,
                    title= 'Regresion lineal')
    st.plotly_chart(figure1)

    X = numeric_df8[x_selected].values.reshape(-1, 1)  
    y = numeric_df8[y_selected].values
    model = LinearRegression()
    model.fit(X, y)
    coef_Deter = model.score(X, y)
    coef_Correl = np.sqrt(coef_Deter)
    st.write(f"**Coeficiente de correlación (R):** {coef_Correl:.4f}")

#Regresion lineal multiple
if Frames == "Regresion lineal multiple":
    st.title("REGRESION LINEAL MULTIPLE")
    st.header("Selecciona las variables")
    st.subheader("Nota: El grafico no se desplegara hasta que selecciones las variables independientes")
    x_selected2 = st.sidebar.multiselect(label = "x (Variables independientes)", options= numeric_cols8)
    y_selected2 = st.sidebar.selectbox(label = "y (Variable dependiente)", options= numeric_cols8)
    if x_selected2 and y_selected2:
        figure2= px.scatter(data_frame=numeric_df8, x=x_selected2, y= y_selected2,
                    title= 'Regresion lineal multiple')
        st.plotly_chart(figure2)
        
        X2 = numeric_df8[x_selected2].values  
        y2 = numeric_df8[y_selected2].values  
        
        model2 = LinearRegression()
        model2.fit(X2, y2)
        
        coef_Deter2 = model2.score(X2, y2)
        
        coef_Correl2 = np.sqrt(coef_Deter2)
        
        st.write(f"**Coeficiente de correlación múltiple (R):** {coef_Correl2:.4f}")

#REGRESION NO LINEAL
def funcion_cuadratica(x, a, b, c):
    return a*x**2 + b*x + c

def funcion_exponencial(x, a, b, c):
    return a * np.exp(b * x) + c

def funcion_inversa(x, a):
    return 1 / (a * x)

def funcion_senoidal(x, a, b):
    return a * np.sin(x) + b

def funcion_tangencial(x, a, b):
    return a * np.tan(x) + b

def funcion_valor_absoluto(x, a, b, c):
    return a * np.abs(x) + b * x + c

def funcion_cociente_polinomios(x, a, b, c):
    return (a * x**2 + b) / (c * x)

def funcion_logaritmica(x, a, b):
    return a * np.log(x) + b

def funcion_lineal_producto_coef(x, a, b, c):
    return a * x + b * x + c * x

def funcion_cuadratica_inversa(x, a):
    return 1 / (a * x**2)

def funcion_polinomial_inversa(x, a, b, c):
    return a / b * x**2 + c * x

if Frames == "Regresion no lineal":
    st.title("REGRESION NO LINEAL")
    st.header("Selecciona el tipo de funcion para hacer el ajuste")

    x_selected3 = st.sidebar.selectbox(label="Variable independiente (X)", options=numeric_cols8)
    y_selected3 = st.sidebar.selectbox(label="Variable dependiente (Y)", options=numeric_cols8)

    funcion_selected = st.sidebar.selectbox("Selecciona la función de ajuste",
        options=["Función cuadrática", "Función exponencial", "Función inversa", "Función senoidal", 
                 "Función tangencial", "Función Valor absoluto", "Función cociente entre polinomios", 
                 "Función logaritmica", "Función lineal con producto de coeficientes", 
                 "Función cuadrática inversa", "Función polinomial inversa"])
    funciones = {
        "Función cuadrática": funcion_cuadratica,
        "Función exponencial": funcion_exponencial,
        "Función inversa": funcion_inversa,
        "Función senoidal": funcion_senoidal,
        "Función tangencial": funcion_tangencial,
        "Función Valor absoluto": funcion_valor_absoluto,
        "Función cociente entre polinomios": funcion_cociente_polinomios,
        "Función logaritmica": funcion_logaritmica,
        "Función lineal con producto de coeficientes": funcion_lineal_producto_coef,
        "Función cuadrática inversa": funcion_cuadratica_inversa,
        "Función polinomial inversa": funcion_polinomial_inversa
    }

    X3 = numeric_df8[x_selected3].values
    y3 = numeric_df8[y_selected3].values
    
    try:
        popt, _ = curve_fit(funciones[funcion_selected], X3, y3)
        
        y_pred3 = funciones[funcion_selected](X3, *popt)
        
        fig = px.scatter(x=X3, y=y3, title=f'Ajuste de {funcion_selected}', labels={'x': x_selected3, 'y': y_selected3})
        fig.add_scatter(x=X3, y=y_pred3, mode='lines', name=f'Ajuste {funcion_selected}', line = dict(color='red'))
        st.plotly_chart(fig)
        
        correlacion3 = np.corrcoef(y3, y_pred3)[0, 1]
        
        st.write(f"**Coeficiente de correlación (R):** {correlacion3:.4f}")
    
    except Exception as e:
        st.write(f"Error al ajustar la función: {str(e)}")

#REGRESION LOGISTICA
if Frames == "Regresion logistica":
    st.title("REGRESION LOGISTICA")
    st.header("Selecciona los datos para ver")

    binary_cols = [col for col in df3.columns if df3[col].nunique() == 2]

    y_selected = st.sidebar.selectbox("Selecciona la variable dependiente (y, dicotómica)", options=binary_cols)

    x_selected = st.sidebar.multiselect("Selecciona las variables independientes (X)", options=numeric_cols9)

    if x_selected and y_selected:
        X = numeric_df9[x_selected]
        y = df3[y_selected]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        st.write("Matriz de confusión:")
        st.write(cm)

        positive_label = y_test.unique()[1]  

        precision = precision_score(y_test, y_pred, pos_label=positive_label)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, pos_label=positive_label)
        f1 = f1_score(y_test, y_pred, pos_label=positive_label)

        st.write(f"**Precisión:** {precision:.4f}")
        st.write(f"**Exactitud:** {accuracy:.4f}")
        st.write(f"**Sensibilidad:** {recall:.4f}")
        st.write(f"**F1:** {f1:.4f}")

#OTRAS GRAFICAS
if Frames == "Otras graficas":
    st.title("Graficos extra")
    st.header("Selecciona el tipo de grafico que quieres desplegar")
    st.subheader("Con el pandel de seleccion puedes controlar la informacion mostrada en los graficos")
    
    x_var = st.sidebar.selectbox("Selecciona la variable para el eje X", options=numeric_cols8)
    y_var = st.sidebar.selectbox("Selecciona la variable para el eje Y", options=numeric_cols8)
    
    single_var = st.sidebar.selectbox("Selecciona la variable para gráficas de una sola variable", options=numeric_cols8)

    # LINEPLOT
    if st.button("MOSTRAR LINEPLOT"):
        st.subheader("Lineplot")
        fig, ax = plt.subplots()
        sns.lineplot(x=numeric_df8[x_var], y=numeric_df8[y_var], ax=ax)
        ax.set_title("Lineplot")
        st.pyplot(fig)
    
    # SCATTERPLOT
    if st.button("MOSTRAR SCATTERPLOT"):
        st.subheader("Scatterplot")
        fig, ax = plt.subplots()
        sns.scatterplot(x=numeric_df8[x_var], y=numeric_df8[y_var], ax=ax)
        ax.set_title("Scatterplot")
        st.pyplot(fig)

    # PIEPLOT
    if st.button("MOSTRAR PIEPLOT"):
        st.subheader("Pieplot")
        pie_data = numeric_df8[single_var].value_counts()
        fig, ax = plt.subplots()
        ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
        ax.set_title("Pieplot")
        st.pyplot(fig)
    
    # BARPLOT
    if st.button("MOSTRAR BARPLOT"):
        st.subheader("Barplot")
        fig, ax = plt.subplots()
        sns.barplot(x=numeric_df8[single_var].value_counts().index, y=numeric_df8[single_var].value_counts().values, ax=ax)
        ax.set_title("Barplot")
        st.pyplot(fig)

    # HEATMAP
    if st.button("MOSTRAR HEATMAP"):
        st.subheader("Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df8.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Heatmap de la correlación")
        st.pyplot(fig)

    # BOXPLOT
    if st.button("MOSTRAR BOXPLOT"):
        st.subheader("Boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(x=numeric_df8[single_var], ax=ax)
        ax.set_title("Boxplot")
        st.pyplot(fig)
