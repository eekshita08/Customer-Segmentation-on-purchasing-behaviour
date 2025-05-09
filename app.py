import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.express as px

st.markdown("""
## How to Use:
1. **Upload the Dataset**: Start by uploading your RFM (Recency, Frequency, Monetary) dataset in CSV format.
2. **Select Clustering Algorithm**: Choose between KMeans, Agglomerative Clustering, or DBSCAN to segment your customers.
3. **Select Number of Clusters**: If you're using KMeans or Agglomerative Clustering, choose the desired number of clusters (K) using the slider.
4. **View Results**: After running the clustering, you'll see:
   - Clustering Evaluation Metrics (Silhouette Score, Davies-Bouldin Score, Calinski-Harabasz Score)
   - Customer Segmentation summary
   - Cluster Visualization (Interactive Scatter Plot & Heatmap of correlations)
5. **Dataset Requirements**: The dataset must contain at least the following columns:
   - `Recency`: Time since the last purchase
   - `Frequency`: Number of purchases
   - `Monetary`: Total monetary value of purchases
""")

# --- Load Data ---
uploaded_file = st.file_uploader("Upload RFM data (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # Display data
    if st.checkbox("Show raw data"):
        st.write(df.head())
       

    # --- Cluster Selection ---
    algorithm = st.selectbox("Select Clustering Algorithm", ("KMeans", "Agglomerative Clustering", "DBSCAN"))
    
    if algorithm != "DBSCAN":  # For DBSCAN, the number of clusters (K) is not needed
        k = st.slider("Select number of clusters (K)", 2, 10, 3)

    # --- Feature Columns ---
    features = ['Recency', 'Frequency', 'Monetary']

    metrics_dict = {}  # Store metrics for comparison

    # --- Apply Clustering Algorithm ---
    if algorithm == "KMeans":
        if st.button("Run KMeans Clustering"):
            kmeans = KMeans(n_clusters=k, random_state=42)
            df['Cluster'] = kmeans.fit_predict(df[features])
            silhouette = silhouette_score(df[features], df['Cluster'])
            db_score = davies_bouldin_score(df[features], df['Cluster'])
            ch_score = calinski_harabasz_score(df[features], df['Cluster'])
            metrics_dict['KMeans'] = {'Silhouette': silhouette, 'Davies-Bouldin': db_score, 'Calinski-Harabasz': ch_score}
    
    elif algorithm == "Agglomerative Clustering":
        if st.button("Run Agglomerative Clustering"):
            agglom = AgglomerativeClustering(n_clusters=k)
            df['Cluster'] = agglom.fit_predict(df[features])
            silhouette = silhouette_score(df[features], df['Cluster'])
            db_score = davies_bouldin_score(df[features], df['Cluster'])
            ch_score = calinski_harabasz_score(df[features], df['Cluster'])
            metrics_dict['Agglomerative'] = {'Silhouette': silhouette, 'Davies-Bouldin': db_score, 'Calinski-Harabasz': ch_score}
    
    elif algorithm == "DBSCAN":
        if st.button("Run DBSCAN Clustering"):
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            df['Cluster'] = dbscan.fit_predict(df[features])
            silhouette = silhouette_score(df[features], df['Cluster'])
            db_score = davies_bouldin_score(df[features], df['Cluster'])
            ch_score = calinski_harabasz_score(df[features], df['Cluster'])
            metrics_dict['DBSCAN'] = {'Silhouette': silhouette, 'Davies-Bouldin': db_score, 'Calinski-Harabasz': ch_score}

    # --- Display Metrics ---
    if metrics_dict:
        st.subheader("Clustering Evaluation Metrics Comparison")
        metrics_df = pd.DataFrame(metrics_dict).T
        st.dataframe(metrics_df)

        # --- Display Cluster Counts ---
        st.subheader("Customer Segmentation")
        st.write(df['Cluster'].value_counts().rename("Count").reset_index().rename(columns={'index': 'Cluster'}))

        # --- Advanced Visualizations ---
        # Heatmap of correlations between features
        st.subheader("Correlation Heatmap")
        corr_matrix = df[features].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Interactive Scatter Plot for Customer Segmentation
        st.subheader("Interactive Cluster Visualization")
        fig = px.scatter(df, x='Recency', y='Monetary', color='Cluster',
                         title="Customer Segmentation (Interactive)", 
                         labels={'Recency': 'Recency', 'Monetary': 'Monetary'})
        st.plotly_chart(fig)

        # --- Show Average RFM by Cluster ---
        st.subheader("Average RFM by Cluster")
        avg_rfm = df.groupby('Cluster')[features].mean().reset_index()
        st.dataframe(avg_rfm)