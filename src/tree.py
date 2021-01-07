import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
st.title('Iris')
df = pd.read_csv("../data/Breast_cancer_data.csv")
if st.checkbox('Show dataframe'):
    st.write(df)

features= df[['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area','mean_smoothness']].values
labels = df['diagnosis'].values
X_train,X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)
alg = ['Decision Tree', 'Support Vector Machine']
classifier = st.selectbox('Which algorithm?', alg)
if classifier=='Decision Tree':
    # dtc = DecisionTreeClassifier()
    # dtc.fit(X_train, y_train)
    # acc = dtc.score(X_test, y_test)
    # st.write('Accuracy: ', acc)
    # pred_dtc = dtc.predict(X_test)
    # cm_dtc=confusion_matrix(y_test,pred_dtc)
    # st.write('Confusion matrix: ', cm_dtc)
    arbreFirst = DecisionTreeClassifier(min_samples_split=30, min_samples_leaf=10)
    arbreFirst.fit(X=X_train.iloc[:, :-1], y=y_train.diagnosis)
    plt.figure(figsize=(90, 90))
    tree = plot_tree(arbreFirst, feature_names=list(df.columns[:-1]), filled=True)
    plot_tree(arbreFirst, feature_names=list(df.columns[:-1]), filled=True)
    st.pyplot(plt)

elif classifier == 'Support Vector Machine':
    svm=SVC()
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_svm = svm.predict(X_test)
    cm=confusion_matrix(y_test,pred_svm)
    st.write('Confusion matrix: ', cm)