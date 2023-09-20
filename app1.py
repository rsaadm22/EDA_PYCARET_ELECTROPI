import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from pycaret.classification import *
from pycaret.regression import *

# Initialize best_model to None at the beginning of the script
best_model = None

# Function to load data
def load_data(file):
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        data = pd.read_excel(file)
    else:
        st.warning("Unsupported file format. Please use CSV or Excel files.")
        return None
    return data

# Function to visualize data
def visualize_data(data, column1, column2, plot_type):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(15, 5))
    if plot_type == 'histogram':
        plt.hist(data[column1], bins=20, color='blue', alpha=0.7, label=column1)
        plt.hist(data[column2], bins=20, color='orange', alpha=0.7, label=column2)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {column1} and {column2}')
        plt.legend()
    elif plot_type == 'boxplot':
        sns.boxplot(data=data[[column1, column2]])
        plt.title(f'Box Plot of {column1} and {column2}')
    elif plot_type == 'scatter':
        sns.scatterplot(x=data[column1], y=data[column2])
        plt.title(f'Scatter Plot of {column1} vs {column2}')
    elif plot_type == 'heatmap':
        heatmap_data = data.pivot_table(index=column1, columns=column2, values='Total', aggfunc='mean')
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm')
        plt.title(f'Heatmap between {column1} and {column2}')
    elif plot_type == 'pie':
        plt.subplot(1, 2, 1)
        data[column1].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(f'Pie Chart of {column1}')
        plt.subplot(1, 2, 2)
        data[column2].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(f'Pie Chart of {column2}')
    else:
        st.warning("Invalid plot type")

    plt.tight_layout()
    st.pyplot()

# Streamlit App
st.title("EDA APP _ELECTRO_PI")

# File Upload
file = st.file_uploader("Upload your data file (CSV or Excel)", type=['csv', 'xlsx'])

if file:
    data = load_data(file)
    if data is not None:
        st.write("### Data Preview")
        st.write(data.head())

        st.write("### Data Information")
        column_table = PrettyTable()
        column_table.field_names = ["Column Name", "Data Type"]
        for column in data.columns:
            column_table.add_row([column, data[column].dtype])
        st.text(column_table)

        st.write("### Data Visualization")
        column1 = st.selectbox("Select the first column", data.columns)
        column2 = st.selectbox("Select the second column", data.columns)
        plot_type = st.selectbox("Select plot type", ['histogram', 'boxplot', 'scatter', 'heatmap', 'pie'])
        visualize_data(data, column1, column2, plot_type)

        st.write("### Pycaret Modeling")
        task = st.selectbox("Select Task", ["None", "Classification", "Regression"])
        if task != "None":
            api_type = st.selectbox("Select API Type", ["Functional API", "OOP API"])
            target = st.selectbox("Select Target Variable", data.columns)
           
            if task == "Classification":
                if api_type == "Functional API":
                    s = setup(data, target=target, session_id=123)
                else:
                    from pycaret.classification import ClassificationExperiment
                    exp = ClassificationExperiment()
                    s = exp.setup(data, target=target, session_id=123)
            elif task == "Regression":
                if api_type == "Functional API":
                    s = setup(data, target=target, session_id=123)
                else:
                    from pycaret.regression import RegressionExperiment
                    exp = RegressionExperiment()
                    s = exp.setup(data, target=target, session_id=123)

            if st.button("Compare Models"):
                new_best_model = None  # Initialize a new variable to store the best model
                if api_type == "Functional API":
                    new_best_model = compare_models()
                else:  # OOP API
                    new_best_model = exp.compare_models()
                best_model = new_best_model  # Update the best_model variable

                # Display PyCaret compare_models result
                st.write("### Compare Models Result")
                st.write(new_best_model)

            analysis_type = st.selectbox("Select Analysis Type", ["None", "Evaluate", "Plot Residuals", "Plot Feature Importance"])
            if analysis_type != "None":
                if analysis_type == "Evaluate":
                    evaluate_model(best_model)
                    # Display PyCaret evaluate_model result (if needed)
                    st.write("### Evaluate Model Result")
                    st.write("Check the plots above.")
                elif analysis_type == "Plot Residuals":
                    plot_model(best_model, plot='residuals')
                    # Display PyCaret plot_model result (if needed)
                    st.write("### Plot Residuals Result")
                    st.write("Check the plot above.")