import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load data
st.title("Customer Segmentation Dashboard")
uploaded_file = st.file_uploader("rfm_data.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    
    # Display data
    if st.checkbox("Show raw data"):
        st.write(df.head())

    # Cluster selection
    k = st.slider("Select number of clusters (K)", 2, 10, 3)

    # Apply KMeans
    if st.button("Run KMeans Clustering"):
        features = ['Recency', 'Frequency', 'Monetary']
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df[features])

        # Display cluster counts
        st.subheader("Customer Segmentation")
        st.write(df['Cluster'].value_counts().rename("Count").reset_index().rename(columns={'index': 'Cluster'}))

        # Plot clusters
        st.subheader("Cluster Visualization")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='Recency', y='Monetary', hue='Cluster', palette='tab10', ax=ax)
        st.pyplot(fig)

        # Show average RFM by cluster
        st.subheader("Average RFM by Cluster")
        avg_rfm = df.groupby('Cluster')[features].mean().reset_index()
        st.dataframe(avg_rfm)
