import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)

#Load the pickled model
model = pickle.load(open('k_means_cluster.pkl','rb'))   
dataset= pd.read_csv('Wholesale_Customers_Data.csv')
x = dataset.iloc[:,2:].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

def predict_note_authentication(fresh,milk,grocery,frozen,detergents,delicassen):
  output= model.predict(sc_x.fit_transform([[fresh,milk,grocery,frozen,detergents,delicassen]]))
  print("cluster number", output)
  if output==[0]:
    prediction="Customer is careless"
  elif output==[1]:
    prediction="Customer is standard"
  elif output==[2]:
    prediction="Customer is target"
  elif output==[3]:
    prediction="Customer is careful"
  else:
    prediction="Customer is sensible" 
  print(prediction)
  return prediction

  
def main():    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning Lab</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Customer Segmenation on Wholesale data ")
    
    fresh = st.number_input("Insert Fresh value")
    milk = st.number_input('Insert Milk value')
    grocery = st.number_input('Insert Grocery value')
    frozen = st.number_input("Insert Frozen value")
    detergents = st.number_input('Insert Detergents value')
    delicassen = st.number_input('Insert Delicassen value')

    resul=""
    if st.button("Prediction"):
      result=predict_note_authentication(fresh,milk,grocery,frozen,detergents,delicassen)
      st.success('Model has predicted: {}'.format(result))  
    if st.button("About"):
      st.header("Developed by Jatin Tak")
      st.subheader("Student, Department of Computer Engineering")
    html_temp = """
    <div class="" style="background-color:orange;" >
    <div class="clearfix">           
    <div class="col-md-12">
    <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning Assignment: K-Means Clustering</p></center> 
    </div>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

if __name__=='__main__':
  main()