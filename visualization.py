import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the visualization function
def visualize_diabetes_data(data):
    # Pair Plot
    st.write("### Pair Plot")
    pair_plot = sns.pairplot(data, hue='Outcome', palette='husl')
    st.pyplot(pair_plot)
    
    # Age Distribution by Outcome
    st.write("### Age Distribution by Outcome")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Outcome', y='Age', data=data, palette='Set2')
    st.pyplot()
    
    # Glucose Level Distribution by Outcome
    st.write("### Glucose Level Distribution by Outcome")
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Outcome', y='Glucose', data=data, palette='Set3')
    st.pyplot()

    # BMI Distribution
    st.write("### BMI Distribution")
    plt.figure(figsize=(8, 6))
    sns.histplot(data['BMI'], bins=20, kde=True, color='green')
    st.pyplot()

def main():
    # Set page title and icon
    st.set_page_config(page_title="Diabetes Prediction App", page_icon=":hospital:")
    st.markdown('<p style="text-align:center;"><img src="https://editor.analyticsvidhya.com/uploads/30738medtec-futuristic-650.jpg" width="400"/></p>', unsafe_allow_html=True)

    # Load data
    data = pd.read_csv("C:\\Users\\shyam\\Downloads\\diabetes.csv")

    # Sidebar title
    st.sidebar.title("Diabetes Prediction")

    # Sidebar options
    option = st.sidebar.selectbox("Select Option", ["Exploratory Data Analysis", "Data Visualization", "Prediction"])

    # Perform actions based on selected option
    if option == "Exploratory Data Analysis":
        # Perform EDA
        # perform_eda(data)
        pass
    elif option == "Data Visualization":
        # Visualize Diabetes Risk Factors
        diabetes_risk_factors = ['Obesity', 'High Blood Pressure', 'High Glucose', 'High Cholesterol']
        visualize_diabetes_risk_factors(diabetes_risk_factors)
        
        # Visualize diabetes data
        visualize_diabetes_data(data)
    elif option == "Prediction":
        # Perform prediction
        pass

if __name__ == "__main__":
    main()
