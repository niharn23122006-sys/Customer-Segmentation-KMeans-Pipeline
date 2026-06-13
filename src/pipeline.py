"""
Customer Segmentation using Unsupervised Learning (K-Means Clustering)
=========================================================================
Performs customer segmentation analysis to identify distinct customer groups
based on purchasing behavior and demographic characteristics.

Features:
- Dynamic dataset fetching via kagglehub
- Comprehensive data cleaning and feature selection
- Elbow Method for optimal K determination
- Silhouette Score analysis for cluster validation
- K-Means clustering with optimal parameters
- Detailed cluster profiling and behavior analysis
- Production-ready 3D Cluster Visualization for presentations/social media
"""

import os
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, List
import matplotlib.pyplot as plt
from collections import defaultdict

import kagglehub
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# Ensure local outputs directory exists
os.makedirs('outputs', exist_ok=True)


# =====================================================================
# 1. DATA LOADING
# =====================================================================

def fetch_dataset() -> Path:
    """
    Fetch the Customer Segmentation dataset from Kaggle using kagglehub.
    """
    print("📥 Fetching Customer Segmentation dataset from Kaggle...")
    path = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")
    print(f"✓ Dataset downloaded to: {path}\n")
    return Path(path)


def find_csv_files(dataset_path: Path) -> List[Path]:
    """
    Locate all CSV files in the dataset directory.
    """
    return list(dataset_path.glob('*.csv'))


def load_data(dataset_path: Path) -> pd.DataFrame:
    """
    Load the customer data from the dataset directory.
    """
    csv_files = find_csv_files(dataset_path)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")

    data_file = csv_files[0]
    print(f"✓ Loading data from: {data_file.name}")

    df = pd.read_csv(data_file)
    print(f"✓ Data shape: {df.shape}")
    print(f"✓ Columns: {', '.join(df.columns.tolist())}\n")

    return df


# =====================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================

def perform_eda(df: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis on the customer dataset.
    """
    print("="*70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)

    print(f"\n📊 Dataset Info:")
    print(f"  - Total records: {len(df)}")
    print(f"  - Total features: {len(df.columns)}")
    print(f"  - Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

    print(f"\n📋 Feature Overview:")
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100 if null_count > 0 else 0
        print(f"  {col:25} {str(dtype):10} Non-null: {non_null:6} Missing: {null_count:5} ({null_pct:5.1f}%)")

    print(f"\n🔢 Numeric Features Statistics:")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    print(numeric_df.describe().to_string())


# =====================================================================
# 3. DATA CLEANING
# =====================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the customer dataset by handling missing values and IDs.
    """
    print("\n" + "="*70)
    print("DATA CLEANING")
    print("="*70)

    initial_shape = df.shape

    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        print(f"✓ Removed rows with missing values")

    print(f"✓ Shape before cleaning: {initial_shape}")
    print(f"✓ Shape after cleaning: {df.shape}")

    id_cols = [col for col in df.columns if 'id' in col.lower() or 'customer' in col.lower()]
    if id_cols:
        print(f"✓ Identified ID columns: {id_cols}")
        for col in id_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f"  - Removed: {col}")

    return df


def select_numeric_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select numeric features for segmentation.
    """
    print("\n" + "="*70)
    print("FEATURE SELECTION")
    print("="*70)

    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"\n📊 Available numeric features ({len(numeric_features)}):")
    for i, feat in enumerate(numeric_features, 1):
        print(f"  {i}. {feat}")

    X = df[numeric_features].copy()
    print(f"\n✓ Selected {len(numeric_features)} numeric features for clustering")
    return X, numeric_features


# =====================================================================
# 4. FEATURE SCALING
# =====================================================================

def scale_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """
    Standardize features uniformly using StandardScaler.
    """
    print("\n" + "="*70)
    print("FEATURE SCALING")
    print("="*70)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    print(f"\n✓ Features standardized using StandardScaler")
    return X_scaled, scaler, X_scaled_df


# =====================================================================
# 5. ELBOW METHOD
# =====================================================================

def elbow_method(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> Dict[int, float]:
    """
    Calculate WCSS metrics across a range of K values.
    """
    print("\n" + "="*70)
    print("ELBOW METHOD - FINDING OPTIMAL K")
    print("="*70)

    wcss = {}
    print(f"\n🔄 Testing K values from {k_range.start} to {k_range.stop - 1}...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss[k] = kmeans.inertia_
        print(f"  K={k:2d} → WCSS: {kmeans.inertia_:10,.2f}")

    return wcss


# =====================================================================
# 6. SILHOUETTE ANALYSIS
# =====================================================================

def silhouette_analysis(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> Dict[int, float]:
    """
    Perform Silhouette analysis to validate structural cluster density quality.
    """
    print("\n" + "="*70)
    print("SILHOUETTE ANALYSIS - CLUSTER VALIDATION")
    print("="*70)

    silhouette_scores = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores[k] = sil_score
        print(f"  K={k:2d} → Silhouette Score: {sil_score:7.4f}")

    best_k_sil = max(silhouette_scores, key=silhouette_scores.get)
    print(f"\n🏆 Best K (by Silhouette Score): {best_k_sil} (Score: {silhouette_scores[best_k_sil]:.4f})")
    return silhouette_scores


def recommend_optimal_k(wcss: Dict[int, float], silhouette_scores: Dict[int, float]) -> int:
    """
    Determine recommended optimal K configuration.
    """
    return max(silhouette_scores, key=silhouette_scores.get)


# =====================================================================
# 7. K-MEANS CLUSTERING
# =====================================================================

def train_kmeans(X_scaled: np.ndarray, optimal_k: int) -> Tuple[KMeans, np.ndarray]:
    """
    Train final K-Means algorithm using optimal cluster target values.
    """
    print("\n" + "="*70)
    print(f"TRAINING K-MEANS CLUSTERING (K={optimal_k})")
    print("="*70)

    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\n📊 Cluster Distribution:")
    for cluster_id, count in zip(unique, counts):
        percentage = (count / len(cluster_labels)) * 100
        print(f"  Cluster {cluster_id}: {count:6d} customers ({percentage:5.1f}%)")

    return kmeans, cluster_labels


# =====================================================================
# 8. CLUSTER PROFILING
# =====================================================================

def profile_clusters(X_original: pd.DataFrame, cluster_labels: np.ndarray,
                     feature_names: List[str]) -> Tuple[Dict[int, Dict[str, Any]], pd.DataFrame]:
    """
    Profile specific consumer group characteristics.
    """
    df_with_clusters = X_original.copy()
    df_with_clusters['Cluster'] = cluster_labels

    cluster_profiles = {}
    for cluster_id in sorted(df_with_clusters['Cluster'].unique()):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        profile = {
            'size': len(cluster_data),
            'percentage': (len(cluster_data) / len(df_with_clusters)) * 100,
            'statistics': {}
        }
        for feature in feature_names:
            feature_data = cluster_data[feature]
            profile['statistics'][feature] = {
                'mean': feature_data.mean(),
                'median': feature_data.median(),
                'std': feature_data.std(),
                'min': feature_data.min(),
                'max': feature_data.max()
            }
        cluster_profiles[cluster_id] = profile

    return cluster_profiles, df_with_clusters


def generate_cluster_insights(cluster_profiles: Dict[int, Dict[str, Any]], feature_names: List[str]) -> None:
    """
    Print clear marketing strategy business persona breakdowns.
    """
    print("\n" + "="*70)
    print("CLUSTER INSIGHTS & BUSINESS INTERPRETATION")
    print("="*70)

    income_col = next((f for f in feature_names if 'income' in f.lower()), feature_names[0])
    spending_col = next((f for f in feature_names if 'spending' in f.lower() or 'score' in f.lower()), feature_names[1])

    all_avg_incomes = [prof['statistics'][income_col]['mean'] for prof in cluster_profiles.values()]
    global_income_median = np.median(all_avg_incomes)

    for cluster_id, profile in cluster_profiles.items():
        stats = profile['statistics']
        avg_income = stats[income_col]['mean']
        avg_spending = stats[spending_col]['mean']

        income_level = 'High' if avg_income >= global_income_median else 'Low'
        if avg_spending > 60:
            value = "High-Value Target"
        elif avg_spending > 35:
            value = "Medium-Value Target"
        else:
            value = "Low-Value Retention Target"

        print(f"🔍 CLUSTER {cluster_id} -> Profile: Income: {income_level} | Behavior Segments: {value} (Avg Spending: {avg_spending:.1f}/100)")


# =====================================================================
# 9. VISUALIZATION HELPERS
# =====================================================================

def save_elbow_plot(wcss: Dict[int, float], optimal_k: int, output_path: str) -> None:
    plt.figure(figsize=(8, 5))
    k_values = sorted(wcss.keys())
    wcss_values = [wcss[k] for k in k_values]
    plt.plot(k_values, wcss_values, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal K', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_silhouette_plot(silhouette_scores: Dict[int, float], optimal_k: int, output_path: str) -> None:
    plt.figure(figsize=(8, 5))
    k_values = sorted(silhouette_scores.keys())
    sil_values = [silhouette_scores[k] for k in k_values]
    plt.plot(k_values, sil_values, 'go-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Analysis', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_3d_cluster_plot(df_clustered: pd.DataFrame, feature_names: List[str], output_path: str) -> None:
    try:
        age_col = next((f for f in feature_names if 'age' in f.lower()), feature_names[0])
        income_col = next((f for f in feature_names if 'income' in f.lower()), feature_names[1])
        spending_col = next((f for f in feature_names if 'spending' in f.lower() or 'score' in f.lower()), feature_names[2])

        fig = plt.figure(figsize=(10, 8), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')

        clusters = sorted(df_clustered['Cluster'].unique())
        colors = ['#FF5A5F', '#00A699', '#3FC1C0', '#754F44', '#F5A623', '#4A90E2']

        for idx, cluster_id in enumerate(clusters):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
            ax.scatter(
                cluster_data[age_col], cluster_data[income_col], cluster_data[spending_col],
                c=colors[idx % len(colors)], label=f'Cluster {cluster_id}', s=60, alpha=0.8, edgecolors='w'
            )

        ax.set_xlabel(age_col, fontweight='bold')
        ax.set_ylabel(income_col, fontweight='bold')
        ax.set_zlabel(spending_col, fontweight='bold')
        plt.title('Customer Segmentation Analysis (K-Means Clustering)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"⚠️ Failed to generate 3D plot: {e}")


# =====================================================================
# 10. MAIN PIPELINE
# =====================================================================

def main():
    try:
        dataset_path = fetch_dataset()
        df = load_data(dataset_path)
        perform_eda(df)
        df_clean = clean_data(df)
        X, feature_names = select_numeric_features(df_clean)
        X_scaled, scaler, X_scaled_df = scale_features(X)

        wcss = elbow_method(X_scaled, k_range=range(2, 11))
        silhouette_scores = silhouette_analysis(X_scaled, k_range=range(2, 11))
        optimal_k = recommend_optimal_k(wcss, silhouette_scores)

        kmeans_model, cluster_labels = train_kmeans(X_scaled, optimal_k)
        cluster_profiles, df_clustered = profile_clusters(X, cluster_labels, feature_names)
        generate_cluster_insights(cluster_profiles, feature_names)

        # FIX: Standard relative output target structure paths
        output_csv = 'outputs/customer_segments.csv'
        elbow_path = 'outputs/elbow_method.png'
        silhouette_path = 'outputs/silhouette_analysis.png'
        cluster_3d_path = 'outputs/customer_clusters_3d.png'

        save_elbow_plot(wcss, optimal_k, elbow_path)
        save_silhouette_plot(silhouette_scores, optimal_k, silhouette_path)
        save_3d_cluster_plot(df_clustered, feature_names, cluster_3d_path)

        df_clustered.to_csv(output_csv, index=False)
        print("\n✓ PIPELINE COMPLETED SUCCESSFULLY - Visual assets saved to local outputs/ folder!")

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
