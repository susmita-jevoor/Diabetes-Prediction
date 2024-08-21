import streamlit as st
import pandas as pd
from data_prep1 import load_data
from eda1 import perform_eda
from modeling import data_preprocessing, model_building, predict_diabetes
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from data_prep1 import load_data


def main():
    # Set page title and icon
    st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺")
    st.markdown('<p style="text-align:center;"><img src="https://editor.analyticsvidhya.com/uploads/30738medtec-futuristic-650.jpg" width="400"/></p>', unsafe_allow_html=True)

    # Load data
    data = load_data("C:\\Users\\shyam\\Downloads\\diabetes.csv")

    # Sidebar title
    st.sidebar.title("Diabetes Prediction")

    # Sidebar options
    option = st.sidebar.selectbox("Select Option", ["Exploratory Data Analysis", "Data Visualization", "Prediction"])

    # Perform actions based on selected option
    if option == "Exploratory Data Analysis":
        perform_eda(data)
    elif option == "Data Visualization":
        st.subheader("Visualize Diabetes Risk Factors")
        diabetes_risk_factors = ['Obesity', 'High Blood Pressure', 'High Glucose', 'High Cholesterol']

        # Visualize Diabetes Risk Factors
        st.write("### Pair Plot")
        pair_plot = sns.pairplot(data, hue='Outcome', palette='husl')
        st.pyplot(pair_plot)

        st.write("### Age Distribution by Outcome")
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Outcome', y='Age', data=data, palette='Set2')
        st.pyplot()

        # Glucose Level Distribution by Outcome
        st.write("### Glucose Level Distribution by Outcome")
        plt.figure(figsize=(8, 6))
        sns.histplot(x='Glucose', hue='Outcome', data=data, bins=20, kde=True, palette='Set3')
        st.pyplot()

        st.write("### BMI Distribution")
        plt.figure(figsize=(8, 6))
        sns.histplot(data['BMI'], bins=20, kde=True, color='green')
        st.pyplot()

    elif option == "Prediction":
        X_train, X_test, y_train, y_test, scaler = data_preprocessing(data)
        model = model_building(X_train, y_train)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Making prediction
        prediction = predict_diabetes(model, X_test)
        
        # Displaying results
        st.subheader("Prediction Results")
        st.write("Accuracy: ", accuracy_score(y_test, prediction))

        # Confusion matrix
        cm = confusion_matrix(y_test, prediction)
        st.subheader("Confusion Matrix")
        st.write(cm)

        # Classification report
        st.subheader("Classification Report")
        st.write(classification_report(y_test, prediction))

        # Visualize predicted outcomes vs actual outcomes
        st.subheader("Visualize Predicted Outcomes vs Actual Outcomes")
        df_results = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})
        st.write(df_results)

        # Plot predicted vs actual outcomes
        sns.countplot(x='Actual', hue='Predicted', data=df_results)
        st.pyplot()

if __name__ == "__main__":
    main()
