import os
import sys
sys.path.append(os.getcwd())
import time
import joblib
from matplotlib.font_manager import FontProperties
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from Kit import data_clean
from sklearn.mixture import GaussianMixture

# Reading CSV file
file_path = r'./exp_1/release_data.csv'
df = pd.read_csv(file_path)

# Extract feature column
feature_columns = ['Delta', 'Theta', 'LowAlpha', 'HighAlpha', 'LowBeta', 'HighBeta', 'LowGamma', 'MiddleGamma', 'HRV']
features = df[feature_columns].values

# Normalization to the square root operation
sqrt_features = np.sqrt(features[:, :-1])

# Combined feature vector
merge_feature_vector = np.column_stack((sqrt_features, features[:, -1]))

# Outlier removal
outliers_mask = data_clean.detect_outliers_knn(merge_feature_vector,df['Level'])
# outliers_mask = CleanDataKit.detect_outliers_2dshow(merge_feature_vector, df['Level'], n_clusters=3, threshold_percentile=90,show=True)

# Delete outlier
cleaned_data = merge_feature_vector[~outliers_mask]
y_cleaned = df['Level'][~outliers_mask]

# The expanded data set is divided into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cleaned_data, y_cleaned, test_size=0.2, random_state=42)


# Maps the label to the corresponding category name
class_labels = ["baseline", "low", "high"]

# Use a random forest for classification and calculate the execution time
rf_classifier = RandomForestClassifier(n_estimators=400, random_state=24, max_features='sqrt', n_jobs=-1, oob_score=True)
rf_classifier.fit(X_train, y_train)
start_time_rf = time.time()
y_pred_rf = rf_classifier.predict(X_test)

# Calculate test set accuracy
accuracy_test_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random forest test set accuracy: {accuracy_test_rf}')
end_time_rf = time.time()

# Calculate the outside pocket error
oob_error = 1 - rf_classifier.oob_score_
print("Out-of-Bag Error:", oob_error)

# Calculate the training set accuracy
y_pred_train_rf = rf_classifier.predict(X_train)
accuracy_train_rf = accuracy_score(y_train, y_pred_train_rf)
print(f'Random forest training set accuracy: {accuracy_train_rf}')

# Gets the parameters of a random forest
rf_params = rf_classifier.get_params()

# Print parameter
print("Random Forest Classifier Parameters:")
for param, value in rf_params.items():
    print(f"{param}: {value}")

# Output confusion matrix and classification report
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf, labels=[0, 1, 2], normalize='true')  # 设置normalize='true'以显示比例
classification_rep_rf = classification_report(y_test, y_pred_rf, target_names=class_labels)
print(f'Random forest confusion matrix:\n{conf_matrix_rf}')
print(f'Random forest classification report:\n{classification_rep_rf}')

# Create a chart with multiple subgraphs
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))

font_properties = FontProperties(fname='Font/msyh.ttf') 

# Visualizing random forest confusion matrix
sns.heatmap(conf_matrix_rf, 
            annot=True, 
            fmt='.2%', 
            cmap='Blues', 
            xticklabels=class_labels, 
            yticklabels=class_labels, 
            ax=axes, 
            annot_kws={'fontproperties':font_properties})
axes.set_title('Random Forest Confusion Matrix')
axes.set_xlabel('Predicted Label')
axes.set_ylabel('True Label')

# Adjust layout
plt.tight_layout()
plt.show()

execution_time_rf = end_time_rf - start_time_rf
print(f'Random forest classifier predicts time：{execution_time_rf} s')

# Save the random forest model
model_filename = 'random_forest_model.joblib'
joblib.dump(rf_classifier, model_filename)
print(f'The random forest model has been saved as: {model_filename}')

# Load the model (optional to ensure successful saving)
loaded_rf_model = joblib.load(model_filename)
y_pred_rf = loaded_rf_model.predict(X_test)

plt.show()

pass