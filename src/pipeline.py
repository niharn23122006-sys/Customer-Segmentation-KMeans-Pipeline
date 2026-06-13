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
from sklearn.metrics import silhouette_score, silhouette_samples

warnings.filterwarnings('ignore')

# Ensure local outputs directory exists safely
os.makedirs('outputs', exist_ok=True)

# =====================================================================
# 1. DATA LOADING
# =====================================================================

def fetch_dataset() -> Path:
    """
    Fetch the Customer Segmentation dataset from Kaggle using kagglehub.

    Returns:
        Path: Directory path containing the dataset files
    """
    print("📥 Fetching Customer Segmentation dataset from Kaggle...")
    path = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")
    print(f"✓ Dataset downloaded to: {path}\n")
    return Path(path)


def find_csv_files(dataset_path: Path) -> List[Path]:
    """
    Locate all CSV files in the dataset directory.

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        List of CSV file paths
    """
    csv_files = list(dataset_path.glob('*.csv'))
    return csv_files


def load_data(dataset_path: Path) -> pd.DataFrame:
    """
    Load the customer data from the dataset directory.
    Automatically finds and loads the appropriate CSV file.

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        DataFrame containing customer data
    """
    csv_files = find_csv_files(dataset_path)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")

    # Use the first CSV file found
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

    Args:
        df: Customer dataset
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

    print(f"\n🔤 Categorical Features:")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"  {col:25} {unique_count:5} unique values")
    else:
        print("  No categorical features found")


# =====================================================================
# 3. DATA CLEANING
# =====================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the customer dataset:
    - Remove rows with missing values
    - Handle outliers (optional)
    - Remove unnecessary ID columns

    Args:
        df: Raw customer dataset

    Returns:
        Cleaned dataset
    """
    print("\n" + "="*70)
    print("DATA CLEANING")
    print("="*70)

    initial_shape = df.shape

    # Remove rows with missing values
    if df.isnull().sum().sum() > 0:
        print(f"\n❌ Missing values detected:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        df = df.dropna()
        print(f"✓ Removed rows with missing values")

    print(f"✓ Shape before cleaning: {initial_shape}")
    print(f"✓ Shape after cleaning: {df.shape}")

    # Remove ID columns (if present) as they're not features
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
    Focus on relevant customer behavior metrics (Income, Spending, Age, etc.)

    Args:
        df: Cleaned customer dataset

    Returns:
        Tuple of (selected_df, feature_names)
    """
    print("\n" + "="*70)
    print("FEATURE SELECTION")
    print("="*70)

    # Select only numeric columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"\n📊 Available numeric features ({len(numeric_features)}):")
    for i, feat in enumerate(numeric_features, 1):
        print(f"  {i}. {feat}")

    # Create a dataframe with selected features
    X = df[numeric_features].copy()

    print(f"\n✓ Selected {len(numeric_features)} numeric features for clustering")
    print(f"✓ Feature matrix shape: {X.shape}")

    return X, numeric_features


# =====================================================================
# 4. FEATURE SCALING
# =====================================================================

def scale_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """
    Standardize features using StandardScaler for clustering.
    Essential for K-Means as it's distance-based.

    Args:
        X: Feature matrix

    Returns:
        Tuple of (scaled_data, scaler, scaled_df)
    """
    print("\n" + "="*70)
    print("FEATURE SCALING")
    print("="*70)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    print(f"\n✓ Features standardized using StandardScaler")
    print(f"  - Mean of scaled features: {X_scaled_df.mean().mean():.6f}")
    print(f"  - Std of scaled features: {X_scaled_df.std().mean():.6f}")

    return X_scaled, scaler, X_scaled_df


# =====================================================================
# 5. ELBOW METHOD
# =====================================================================

def elbow_method(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> Dict[int, float]:
    """
    Implement the Elbow Method to find optimal number of clusters.
    Uses Within-Cluster Sum of Squares (WCSS).

    Args:
        X_scaled: Scaled feature matrix
        k_range: Range of K values to test (default: 2-10)

    Returns:
        Dictionary of {K: WCSS}
    """
    print("\n" + "="*70)
    print("ELBOW METHOD - FINDING OPTIMAL K")
    print("="*70)

    wcss = {}
    inertias = []

    print(f"\n🔄 Testing K values from {k_range.start} to {k_range.stop - 1}...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss[k] = kmeans.inertia_
        inertias.append(kmeans.inertia_)
        print(f"  K={k:2d} → WCSS: {kmeans.inertia_:10,.2f}")

    # Calculate the rate of change (elbow detection)
    print(f"\n📉 Rate of change in WCSS (potential elbow points):")
    for i in range(1, len(inertias)):
        k_val = k_range.start + i
        rate_of_change = inertias[i-1] - inertias[i]
        percent_change = (rate_of_change / inertias[i-1]) * 100
        print(f"  K={k_val}: Δ WCSS = {rate_of_change:10,.2f} ({percent_change:6.2f}%)")

    return wcss


# =====================================================================
# 6. SILHOUETTE ANALYSIS
# =====================================================================

def silhouette_analysis(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> Dict[int, float]:
    """
    Perform Silhouette analysis to validate cluster quality.
    Higher silhouette score indicates better-defined clusters.

    Args:
        X_scaled: Scaled feature matrix
        k_range: Range of K values to test

    Returns:
        Dictionary of {K: silhouette_score}
    """
    print("\n" + "="*70)
    print("SILHOUETTE ANALYSIS - CLUSTER VALIDATION")
    print("="*70)

    silhouette_scores = {}

    print(f"\n📊 Computing Silhouette Scores for K={k_range.start} to {k_range.stop - 1}...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        sil_score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores[k] = sil_score

        print(f"  K={k:2d} → Silhouette Score: {sil_score:7.4f}")

    # Find best K based on silhouette score
    best_k_sil = max(silhouette_scores, key=silhouette_scores.get)
    print(f"\n🏆 Best K (by Silhouette Score): {best_k_sil} (Score: {silhouette_scores[best_k_sil]:.4f})")

    return silhouette_scores


def recommend_optimal_k(wcss: Dict[int, float], silhouette_scores: Dict[int, float]) -> int:
    """
    Recommend optimal K based on Elbow Method and Silhouette Analysis.

    Args:
        wcss: Dictionary of WCSS values
        silhouette_scores: Dictionary of Silhouette scores

    Returns:
        Recommended K value
    """
    print("\n" + "="*70)
    print("OPTIMAL K RECOMMENDATION")
    print("="*70)

    # Method 1: Elbow - look for largest rate of change drop
    wcss_values = sorted(wcss.items())
    rate_changes = []
    for i in range(1, len(wcss_values)):
        k, curr_wcss = wcss_values[i]
        prev_wcss = wcss_values[i-1][1]
        rate = (prev_wcss - curr_wcss) / prev_wcss
        rate_changes.append((k, rate))

    # Find elbow (point where rate of change drops significantly)
    elbow_k = rate_changes[0][0]  # Default to first point
    for i in range(len(rate_changes) - 1):
        if rate_changes[i][1] > 0.1 and rate_changes[i+1][1] < 0.1:
            elbow_k = rate_changes[i+1][0]
            break

    # Method 2: Silhouette - highest score
    best_k_sil = max(silhouette_scores, key=silhouette_scores.get)

    print(f"\n📊 Analysis Summary:")
    print(f"  - Elbow Method suggests: K = {elbow_k}")
    print(f"  - Silhouette Method suggests: K = {best_k_sil}")

    # Use silhouette score as primary metric
    recommended_k = best_k_sil

    print(f"\n🎯 Recommended K: {recommended_k}")
    return recommended_k


# =====================================================================
# 7. K-MEANS CLUSTERING
# =====================================================================

def train_kmeans(X_scaled: np.ndarray, optimal_k: int) -> Tuple[KMeans, np.ndarray]:
    """
    Train the final K-Means model with optimal K.

    Args:
        X_scaled: Scaled feature matrix
        optimal_k: Optimal number of clusters

    Returns:
        Tuple of (trained_model, cluster_labels)
    """
    print("\n" + "="*70)
    print(f"TRAINING K-MEANS CLUSTERING (K={optimal_k})")
    print("="*70)

    print(f"\n🔧 Training K-Means with K={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    print(f"✓ K-Means training completed")
    print(f"✓ Inertia (WCSS): {kmeans.inertia_:,.2f}")

    # Cluster distribution
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
    Create detailed statistical profiles for each cluster.
    Explains distinct purchasing behaviors and characteristics.

    Args:
        X_original: Original (unscaled) feature matrix
        cluster_labels: Cluster assignments for each customer
        feature_names: List of feature names

    Returns:
        Tuple containing dictionary with cluster profiles and the modified DataFrame
    """
    print("\n" + "="*70)
    print("CLUSTER PROFILING & BEHAVIOR ANALYSIS")
    print("="*70)

    # Add cluster labels to the dataframe
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

        # Calculate statistics for each feature
        for feature in feature_names:
            feature_data = cluster_data[feature]
            profile['statistics'][feature] = {
                'mean': feature_data.mean(),
                'std': feature_data.std(),
                'min': feature_data.min(),
                'max': feature_data.max(),
                'median': feature_data.median()
            }

        cluster_profiles[cluster_id] = profile

    return cluster_profiles, df_with_clusters


def print_cluster_profiles(cluster_profiles: Dict[int, Dict[str, Any]]) -> None:
    """
    Print detailed cluster profiles in a readable format.

    Args:
        cluster_profiles: Dictionary of cluster statistics
    """
    print("\n" + "="*70)
    print("DETAILED CLUSTER PROFILES")
    print("="*70)

    for cluster_id, profile in cluster_profiles.items():
        print(f"\n{'='*70}")
        print(f"CLUSTER {cluster_id}")
        print(f"{'='*70}")
        print(f"Size: {profile['size']:,} customers ({profile['percentage']:.1f}%)")
        print(f"\n{'Feature':<25} {'Mean':>12} {'Median':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print("-" * 85)

        for feature, stats in profile['statistics'].items():
            print(f"{feature:<25} {stats['mean']:>12.2f} {stats['median']:>12.2f} "
                  f"{stats['std']:>12.2f} {stats['min']:>12.2f} {stats['max']:>12.2f}")


def generate_cluster_insights(cluster_profiles: Dict[int, Dict[str, Any]],
                              feature_names: List[str]) -> None:
    """
    Generate human-readable business insights for each cluster.

    Args:
        cluster_profiles: Dictionary of cluster statistics
        feature_names: List of feature names
    """
    print("\n" + "="*70)
    print("CLUSTER INSIGHTS & BUSINESS INTERPRETATION")
    print("="*70)

    # Clean non-overlapping extraction heuristics
    age_col = next((f for f in feature_names if 'age' in f.lower()), None)
    income_col = next((f for f in feature_names if 'income' in f.lower()), None)
    spending_col = next((f for f in feature_names if 'spending' in f.lower() or 'score' in f.lower()), None)

    # Use actual cluster profile distributions to determine relative bounds
    all_avg_incomes = [prof['statistics'][income_col]['mean'] for prof in cluster_profiles.values() if income_col]
    global_income_median = np.median(all_avg_incomes) if all_avg_incomes else 50

    for cluster_id, profile in cluster_profiles.items():
        print(f"\n🔍 CLUSTER {cluster_id} - Characteristics:")
        print(f"   Size: {profile['size']:,} customers ({profile['percentage']:.1f}%)\n")

        stats = profile['statistics']

        if age_col:
            avg_age = stats[age_col]['mean']
            print(f"   👥 Average Age: {avg_age:.1f} years old")

        if income_col:
            avg_income = stats[income_col]['mean']
            print(f"   💰 Average Income: ${avg_income:,.2f}k")

            # Income-based segmentation comparison against relative dataset averages
            income_level = 'High' if avg_income >= global_income_median else 'Low'
            print(f"   💸 Income Level: {income_level}")

        if spending_col:
            avg_spending = stats[spending_col]['mean']
            print(f"   🛍️  Average Spending Score: {avg_spending:.2f}/100")

            # Direct assessment on spending behavior definitions
            if avg_spending > 60:
                value = "High-Value"
            elif avg_spending > 35:
                value = "Medium-Value"
            else:
                value = "Low-Value"

            print(f"   ⭐ Customer Segment: {value}")

        print()


# =====================================================================
# 9. VISUALIZATION HELPERS
# =====================================================================

def save_elbow_plot(wcss: Dict[int, float], optimal_k: int, output_path: str = None) -> None:
    """
    Create and save Elbow Method visualization.
    """
    try:
        plt.figure(figsize=(10, 6))
        k_values = sorted(wcss.keys())
        wcss_values = [wcss[k] for k in k_values]

        plt.plot(k_values, wcss_values, 'bo-', linewidth=2, markersize=8)
        plt.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
        plt.xlabel('Number of Clusters (K)', fontsize=12)
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
        plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Elbow plot saved to {output_path}")

        plt.show()  # Display plot inline in notebook/Colab environments
    except Exception as e:
        print(f"⚠️  Could not create elbow plot: {e}")


def save_silhouette_plot(silhouette_scores: Dict[int, float], optimal_k: int, output_path: str = None) -> None:
    """
    Create and save Silhouette Score visualization.
    """
    try:
        plt.figure(figsize=(10, 6))
        k_values = sorted(silhouette_scores.keys())
        sil_values = [silhouette_scores[k] for k in k_values]

        plt.plot(k_values, sil_values, 'go-', linewidth=2, markersize=8)
        plt.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
        plt.xlabel('Number of Clusters (K)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Silhouette Score Analysis', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        plt.legend(fontsize=11)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Silhouette plot saved to {output_path}")

        plt.show()  # Display plot inline in notebook/Colab environments
    except Exception as e:
        print(f"⚠️  Could not create silhouette plot: {e}")


def save_3d_cluster_plot(df_clustered: pd.DataFrame, feature_names: List[str], output_path: str = None) -> None:
    """
    Generates a presentation-ready 3D scatter plot of clusters.
    """
    try:
        # Dynamically locate core target structural axis features
        age_col = next((f for f in feature_names if 'age' in f.lower()), feature_names[0])
        income_col = next((f for f in feature_names if 'income' in f.lower()), feature_names[1])
        spending_col = next((f for f in feature_names if 'spending' in f.lower() or 'score' in f.lower()), feature_names[2])

        fig = plt.figure(figsize=(12, 9), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')

        # High-contrast professional palette for explicit cluster grouping boundaries
        clusters = sorted(df_clustered['Cluster'].unique())
        colors = ['#FF5A5F', '#00A699', '#3FC1C0', '#754F44', '#F5A623', '#4A90E2', '#B8E986']

        for idx, cluster_id in enumerate(clusters):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
            ax.scatter(
                cluster_data[age_col],
                cluster_data[income_col],
                cluster_data[spending_col],
                c=colors[idx % len(colors)],
                label=f'Cluster {cluster_id}',
                s=70,
                alpha=0.85,
                edgecolors='white',
                linewidth=0.5
            )

        # Labels & Aesthetic properties optimized for visual readability
        ax.set_xlabel(f'\n{age_col}', fontsize=11, fontweight='bold', labelpad=10)
        ax.set_ylabel(f'\n{income_col}', fontsize=11, fontweight='bold', labelpad=10)
        ax.set_zlabel(f'\n{spending_col}', fontsize=11, fontweight='bold', labelpad=10)

        plt.title('Customer Segmentation Analysis (K-Means Clustering)\n', fontsize=15, fontweight='bold', pad=20)

        # Grid lines styling
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Legend styling configuration
        ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95), fontsize=10, frameon=True, shadow=True)

        # Set an optimal initial viewing perspective tilt angle
        ax.view_init(elev=25, azim=135)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ 3D Cluster visualization chart saved to {output_path}")

        plt.show()  # Display plot inline in notebook/Colab environments
    except Exception as e:
        print(f"⚠️  Could not create 3D cluster plot: {e}")


# =====================================================================
# 10. MAIN PIPELINE
# =====================================================================

def main():
    """
    Execute the complete customer segmentation pipeline.
    """
    print("\n" + "="*70)
    print("👥 CUSTOMER SEGMENTATION - UNSUPERVISED LEARNING PIPELINE")
    print("="*70)

    try:
        # Step 1: Fetch and Load Data
        dataset_path = fetch_dataset()
        df = load_data(dataset_path)

        # Step 2: EDA
        perform_eda(df)

        # Step 3: Data Cleaning
        df_clean = clean_data(df)

        # Step 4: Feature Selection
        X, feature_names = select_numeric_features(df_clean)

        # Step 5: Feature Scaling
        X_scaled, scaler, X_scaled_df = scale_features(X)

        # Step 6: Elbow Method
        wcss = elbow_method(X_scaled, k_range=range(2, 11))

        # Step 7: Silhouette Analysis
        silhouette_scores = silhouette_analysis(X_scaled, k_range=range(2, 11))

        # Step 8: Determine Optimal K
        optimal_k = recommend_optimal_k(wcss, silhouette_scores)

        # Step 9: Train K-Means with Optimal K
        kmeans_model, cluster_labels = train_kmeans(X_scaled, optimal_k)

        # Step 10: Profile Clusters
        cluster_profiles, df_clustered = profile_clusters(X, cluster_labels, feature_names)

        # Step 11: Print Cluster Profiles
        print_cluster_profiles(cluster_profiles)

        # Step 12: Generate Business Insights
        generate_cluster_insights(cluster_profiles, feature_names)

        # Step 13: Create Visualizations (With verified Target Folder structures)
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)

        # --- FIXED PATHS FOR GOOGLE COLAB AND LOCAL ENVIRONMENTS ---
        output_csv = 'outputs/customer_segments.csv'
        elbow_path = 'outputs/elbow_method.png'
        silhouette_path = 'outputs/silhouette_analysis.png'
        cluster_3d_path = 'outputs/customer_clusters_3d.png'

        save_elbow_plot(wcss, optimal_k, elbow_path)
        save_silhouette_plot(silhouette_scores, optimal_k, silhouette_path)
        save_3d_cluster_plot(df_clustered, feature_names, cluster_3d_path)

        # Save clustered data
        df_clustered.to_csv(output_csv, index=False)
        print(f"✓ Clustered data saved to {output_csv}")

        # Summary Statistics
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)
        print(f"\n✓ Total Customers Analyzed: {len(df_clean):,}")
        print(f"✓ Features Used: {len(feature_names)}")
        print(f"✓ Optimal Number of Clusters: {optimal_k}")
        print(f"✓ Final Silhouette Score: {silhouette_scores[optimal_k]:.4f}")
        print(f"✓ Final Model Inertia: {kmeans_model.inertia_:,.2f}")

        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\n📊 Output Files Generated:")
        print("  1. outputs/customer_segments.csv - Customer data with cluster assignments")
        print("  2. outputs/elbow_method.png - Elbow Method visualization")
        print("  3. outputs/silhouette_analysis.png - Silhouette Score visualization")
        print("  4. outputs/customer_clusters_3d.png - High-impact 3D Data Visual representation")

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
if __name__ == "__main__":
    main()
