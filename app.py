import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a pandas DataFrame
transactions = pd.read_csv('Transactional_data_retail_01.csv')
transactions.columns = transactions.columns.str.strip()

# App layout
st.sidebar.header("Input Options")
stock_code = st.sidebar.selectbox("Select a Stock Code:", transactions['StockCode'].unique())

# Checking the data and displaying columns
st.write("Available columns in the DataFrame:")
st.write(transactions.columns.tolist())

# Handle the case where the DataFrame is empty
if transactions.empty:
    st.error("The DataFrame is empty. Please check the CSV file.")
else:
    st.write(transactions.head())

    # Check if required columns are present in the data
    required_columns = ['StockCode', 'InvoiceDate', 'Quantity']
    missing_columns = [col for col in required_columns if col not in transactions.columns]
    
    if missing_columns:
        st.error(f"Missing columns in the data: {', '.join(missing_columns)}. Please check the CSV file.")
    else:
        # Data Processing: Aggregating sales quantity by date for the selected stock code
        product_sales = transactions[transactions['StockCode'] == stock_code].groupby('InvoiceDate')['Quantity'].sum()
        
        # Plotting the actual vs predicted demand
        st.subheader(f"Demand Forecasting for {stock_code}")
        fig, ax = plt.subplots(figsize=(10, 6))
        product_sales.plot(ax=ax, marker='o', label='Actual Demand')
        
        # Dummy predicted demand data for illustration
        predicted_sales = product_sales + (product_sales * 0.1)  # Adjusting for prediction difference
        predicted_sales.plot(ax=ax, linestyle='--', marker='x', label='Predicted Demand')

        ax.set_xlabel('Date')
        ax.set_ylabel('Demand')
        ax.set_title(f"Actual vs Predicted Demand for {stock_code}")
        ax.legend()
        st.pyplot(fig)

        # Error distribution for training and testing sets
        st.subheader("Error Distribution")

        # Assuming some dummy error data for training and testing sets
        train_error = product_sales - (product_sales * 0.95)
        test_error = product_sales - predicted_sales

        # Plotting error distributions using histograms
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training Error
        sns.histplot(train_error, ax=axes[0], kde=True, color='green', bins=15)
        axes[0].set_title('Training Error Distribution')
        axes[0].set_xlabel('Error')
        
        # Testing Error
        sns.histplot(test_error, ax=axes[1], kde=True, color='red', bins=15)
        axes[1].set_title('Testing Error Distribution')
        axes[1].set_xlabel('Error')
        
        st.pyplot(fig)
