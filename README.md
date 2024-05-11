# Breast Cancer Prediction

This Jupyter notebook focuses on using various machine learning techniques to classify and cluster breast cancer cases. The project employs a wide array of classifiers and clustering methods, detailing their implementation and functionality.

## Dataset

The dataset for this project is sourced from a public breast cancer dataset, including various attributes essential for cancer classification and analysis.

## Preprocessing Steps

The preprocessing involves several key steps to optimize the data:

1. **Removing Unnecessary Columns**: Drops columns like 'id' which are not useful for the analysis.
2. **Handling Missing Values**: Ensures there are no missing values in the dataset that could impair model performance.
3. **Encoding Categorical Labels**: Transforms categorical data into numeric format using `LabelEncoder`.
4. **Feature Scaling**: Applies `StandardScaler` to standardize features, ensuring equal importance is given to all features.
5. **Outlier Detection and Removal**: Employs the Interquartile Range (IQR) method to detect and remove outliers, enhancing model accuracy.

## Modeling Techniques

### Classifiers

- **Support Vector Machine (SVM)**: Implemented using `SVC` from Scikit-learn, optimizing the hyperparameters for kernel type and margin softness.
- **K-Nearest Neighbors (KNN)**: Uses `KNeighborsClassifier`, with the number of neighbors as a variable to tune.
- **Random Forest Classifier**: Utilizes `RandomForestClassifier`, with parameters for the number of trees and depth of the trees configured.
- **Decision Tree Classifier**: Configured using `DecisionTreeClassifier`, with adjustments for the depth of the tree and criteria for splitting.
- **Linear Regression**: Applied using `LinearRegression`, focusing on relationship modeling between features.
- **Logistic Regression**: Implemented with `LogisticRegression`, optimizing for the regularization strength and solver type.
- **Gaussian Naive Bayes**: Uses `GaussianNB`, suitable for distributions in the dataset and requires no configuration.

### Clustering Methods

- **K-Means**: Configured with `KMeans` from Scikit-learn, specifying the number of clusters and initialization method.
- **K-Medoids**: Utilizes the `KMedoids` class from Scikit-learn, focusing on robustness to noise and outliers.
- **DBSCAN**: Implemented using `DBSCAN`, with parameters for minimum samples per cluster and epsilon distance.
- **Agglomerative Clustering**: Uses `AgglomerativeClustering`, determining the number of clusters and linkage criteria.

### Hierarchical Clustering Visualization

- **Dendrogram**: Visualized using Scipy's `dendrogram` function to showcase the hierarchical clustering process.

## Libraries Used

- Pandas
- Scikit-learn
- Seaborn
- Matplotlib

## Installation

To install the required libraries, run the following command:

```bash
pip install pandas scikit-learn seaborn matplotlib
```

## Usage

Run the notebook through Jupyter Notebook or JupyterLab:

```bash
jupyter lab
```

Navigate to the notebook file within Jupyter to view and execute it.

## Conclusion

The notebook offers a comprehensive analysis using multiple machine learning classifiers and clustering methods to predict breast cancer cases effectively. Through detailed preprocessing and methodical application of each technique, the data is thoroughly analyzed, enhancing both the accuracy and reliability of the predictions.

---
