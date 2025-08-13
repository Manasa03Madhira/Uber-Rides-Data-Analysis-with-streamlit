import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit page setup
st.set_page_config(page_title="Uber Data Analysis", layout="wide")
st.title("🚖 Uber Data Analysis Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your Uber dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load dataset
    dataset = pd.read_csv(uploaded_file)
    
    # Preview data
    st.subheader("Data Preview")
    st.dataframe(dataset.head())

    # Dataset info
    st.subheader("Dataset Information")
    st.write(f"**Shape:** {dataset.shape[0]} rows × {dataset.shape[1]} columns")
    st.write("**Column Types:**")
    st.write(dataset.dtypes)

    # Missing values
    st.subheader("Missing Values")
    st.write(dataset.isnull().sum())

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_dataset = dataset.select_dtypes(include=['number'])
    if not numeric_dataset.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            numeric_dataset.corr(),
            cmap='BrBG',
            fmt='.2f',
            linewidths=0.5,
            annot=True,
            annot_kws={"size": 10, "color": "black"}
        )
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for correlation heatmap.")

    # Example categorical count plots
    st.subheader("Categorical Distributions")
    cat_columns = dataset.select_dtypes(include=['object']).columns
    for col in cat_columns:
        if dataset[col].nunique() < 20:  # Only small cardinality for readability
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(x=col, data=dataset, ax=ax)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

else:
    st.info("👆 Upload a CSV file to begin analysis.")
