# 👥 Customer Segmentation Pipeline via K-Means Clustering

An end-to-end unsupervised machine learning solution providing algorithmic consumer grouping using the Kaggle **Mall Customer Segmentation** dataset. 

## 🚀 Pipeline Features
* **Automated Data Ingestion:** Ingests raw structural files dynamically using `kagglehub`.
* **Algorithmic Optimization:** Employs dual-validation systems (**Elbow Method** & **Silhouette Analysis**) to find the ideal structural target value for K.
* **Robust Variable Engineering:** Scales distances uniformly using `StandardScaler` to prevent feature magnitude bias.
* **Business Intelligence Engine:** Auto-profiles customer subsets into human-readable business personas (High-Value, Medium-Value, Low-Value targets).

## 📊 Optimization & Results
The framework tested iterations between K=2 and K=10. Cross-examination of the variance curve inflection against maximum silhouette density pinpointed the optimal breakdown structure.
### 🪐 Final 3D Cluster Mapping
![Customer Segments 3D Visual Map](outputs/customer_clusters_3d.jpg)

All technical validation charts, data assignments, and distribution heatmaps are automatically generated and saved under the `outputs/` directory.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Core Packages:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
