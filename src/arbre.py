import sklearn
import os
import pandas
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
import matplotlib
import lightgbm as lgb
#matplotlib.use( 'tkagg' )

os.chdir(r"C:\Users\ASUS\PycharmProjects\ml\data")
df = pandas.read_csv("Breast_cancer_data.csv")
st.title('Breast cancer')

st.sidebar.header('User Input Parameters')

def user_input_features():
    depth = st.sidebar.slider('Tree depth', 2,7)
    width = st.sidebar.slider('Tree width', 10,80)
    length = st.sidebar.slider('Tree heigth', 10,80)
    font = st.sidebar.slider('Font', 10,40)

    data = {'Tree_depth': depth,
            'Tree_width': width,
            'Tree_heigth': length,
            'Font': font
            }
    features = pd.DataFrame(data, index=[0])
    return features
input = user_input_features()

st.subheader('User Input parameters')
st.write(input)

dfTrain, dfTest = train_test_split(df,test_size=300,random_state=1,stratify=df.diagnosis)
#arbreFirst = DecisionTreeClassifier(min_samples_split=30,min_samples_leaf=10)
arbreFirst = DecisionTreeClassifier(max_depth=input['Tree_depth'][0])
arbreFirst.fit(X = dfTrain.iloc[:,:-1], y = dfTrain.diagnosis)
plt.figure(figsize=(input['Tree_width'][0],input['Tree_heigth'][0]))
plot_tree(arbreFirst,feature_names = list(df.columns[:-1]),filled=True,fontsize=input['Font'][0])
st.pyplot(plt)


