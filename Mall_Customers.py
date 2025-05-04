import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


df = pd.read_csv('Mall_Customers.csv') ## menampilkan data ke streamlit

df.rename(index=str, columns={
    'Annual Income (k$)' : 'Income', 'Spending Score (1-100)' : 'Score'
}, inplace=True)  

X = df.drop(['CustomerID', 'Gender'],  axis=1) ## menghapus colums customers dan age

st.header("isi dataset")
st.write(X)

## menampilkan panah elbow
clusters =[]
for i in range(1,11):
    km =KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)
    
fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)), y=clusters, ax=ax)
ax.set_title('mencari elbow')
ax.set_xlabel('clusters')
ax.set_ylabel('inertia')

# panah elbow
ax.annotate('Possible elbow point', xy=(3, 140000), xytext=(3, 50000),xycoords='data', 
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

ax.annotate('Possible elbow point', xy=(5, 80000), xytext=(5, 150000),xycoords='data', 
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))


elbo_plot = st.pyplot(fig) ## menampilkan panah elbow di streamlit

## membuat slider streamlit untuk menentukan cluster dari 2-10 , dan naik 1
st.sidebar.subheader("Nilai Jumalah K")
clust = st.sidebar.slider("pilih jumlah cluster:", 2,10,3,1)

## proses clustering
def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(X)
    X['Labels'] = kmean.labels_
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x='Income', y='Score', hue='Labels', style='Labels', data=X, palette=sns.color_palette('hls', n_clust))

    for label in X['Labels'].unique():
        plt.annotate(label,
            (X[X['Labels'] == label]['Income'].mean(),
            X[X['Labels'] == label]['Score'].mean()),
            horizontalalignment='center',
            verticalalignment='center',
            size=20,
            weight='bold',
            color='black')    
    
    st.header('Cluster Plot')            
    st.pyplot(fig)
    st.write(X)

k_means(clust)