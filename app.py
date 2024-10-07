import streamlit as st
import pandas as pd
transactions = pd.read_csv('Transactional_data_retail_01.csv')
transactions.columns = transactions.columns.str.strip()
st.write("Available columns in the DataFrame:")
st.write(transactions.columns.tolist())  
if transactions.empty:
    st.error("The DataFrame is empty. Please check the CSV file.")
else:
    st.write(transactions.head())
    stock_code = 'YOUR_STOCK_CODE'  
    required_columns = ['StockCode', 'InvoiceDate', 'Quantity']
    missing_columns = [col for col in required_columns if col not in transactions.columns]
    if missing_columns:
        st.error(f"Missing columns in the data: {', '.join(missing_columns)}. Please check the CSV file.")
    else:
        product_sales = transactions[transactions['StockCode'] == stock_code].groupby('InvoiceDate')['Quantity'].sum()
        st.write(product_sales)  
