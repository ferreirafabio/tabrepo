from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tabrepo.loaders import Paths
import numpy as np


def get_clusters(df, n_clusters=10):
    df = df.groupby(["framework", "dataset"])["rank"].mean().reset_index(drop=False)

    encoder_fw = LabelEncoder()
    encoder_ds = LabelEncoder()
    df.fillna(-100, inplace=True)
    df['framework_encoded'] = encoder_fw.fit_transform(df['framework'])
    df['dataset_encoded'] = encoder_ds.fit_transform(df['dataset'])

    features = df.drop(['framework', 'dataset'], axis=1)
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
    df['cluster'] = kmeans.fit_predict(features)

    return df


def sample_frameworks_from_clusters(df, num_samples_per_cluster=1):
    sampled_frameworks = []
    for cluster in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster]
        sampled_frameworks += cluster_df.sample(n=num_samples_per_cluster)['framework'].tolist()
    return sampled_frameworks


def plot_clusters(df, plot_3d=False, pca_components=2, num_points=5, fontsize=10, text_distance=1):
    pca = PCA(n_components=pca_components)
    reduced_data = pca.fit_transform(df[['framework_encoded', 'dataset_encoded', 'rank']])

    colors = plt.cm.get_cmap('tab20', len(df['cluster'].unique()))

    fig = plt.figure(figsize=(40, 20))
    if plot_3d:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel('Principal Component 3')
    else:
        ax = fig.add_subplot(111)

    for i, cluster in enumerate(df['cluster'].unique()):
        cluster_color = colors(i)
        cluster_data = reduced_data[df['cluster'] == cluster]
        if plot_3d:
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], color=cluster_color,
                       label=f'Cluster {cluster}')
        else:
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], color=cluster_color, label=f'Cluster {cluster}')

        # Annotating points
        cluster_df = df[df['cluster'] == cluster]
        if len(cluster_data) > num_points:
            sampled_indices = np.random.choice(cluster_data.shape[0], num_points, replace=False)
        else:
            sampled_indices = range(cluster_data.shape[0])

        for idx in sampled_indices:
            point = cluster_data[idx]
            framework = cluster_df.iloc[idx]['framework'].split('_')[0]
            dataset = cluster_df.iloc[idx]['dataset']
            text = f'{framework}, {dataset}'
            if plot_3d:
                ax.text(point[0], point[1], point[2], text, fontsize=fontsize, color=cluster_color,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))
            else:
                ax.annotate(text, (point[0], point[1]), fontsize=fontsize, color=cluster_color,
                            xytext=(text_distance, text_distance), textcoords='offset points',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('Framework Clustering Visualization' + (' (3D)' if plot_3d else ' (2D)'))
    plt.legend()
    plt.savefig('cluster_visualization.png')



portfolio_size = 20
n_runs = 1000

load_path = str(Paths.data_root / "results-baseline-comparison" / "D244_F3_C1416" / "train_test_df_rank.pkl")
df = pd.read_pickle(load_path)

clustered_df = get_clusters(df, n_clusters=portfolio_size)

sampled_frameworks = sample_frameworks_from_clusters(clustered_df)
print(sampled_frameworks)

# plot_clusters(clustered_df, plot_3d=False)

frameworks_of_interest = ["CatBoost", "ExtraTrees", "NeuralNetTorch", "RandomForest", "LightGBM", "XGBoost"]
framework_counts = {framework: 0 for framework in frameworks_of_interest}


for _ in range(n_runs):
    clustered_df = get_clusters(df, n_clusters=portfolio_size)

    sampled_frameworks = sample_frameworks_from_clusters(clustered_df)

    for sampled_framework in sampled_frameworks:
        for framework in frameworks_of_interest:
            if sampled_framework.startswith(framework):
                framework_counts[framework] += 1

print(framework_counts)
