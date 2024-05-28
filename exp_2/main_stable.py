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
file_path = r'./exp_2/release_data.csv'
df = pd.read_csv(file_path)

# Load the model (optional to ensure successful saving)
model_filename = r'./exp_1/random_forest_model.joblib'
rf_classifier = joblib.load(model_filename)

# Gets the parameters of a random forest
rf_params = rf_classifier.get_params()

# Print parameter
print("Random Forest Classifier Parameters:")
for param, value in rf_params.items():
    print(f"{param}: {value}")

# Extract feature column
feature_columns = ['Delta', 'Theta', 'LowAlpha', 'HighAlpha', 'LowBeta', 'HighBeta', 'LowGamma', 'MiddleGamma', 'HRV']
features = df[feature_columns].values

# Normalization to the square root operation
sqrt_features = np.sqrt(features[:, :-1])

# Combined feature vector
merge_feature_vector = np.column_stack((sqrt_features, features[:, -1]))
y_pred_rf = rf_classifier.predict(merge_feature_vector)

# Use random forest for classification and calculate execution time
# Build column names for the output CSV file
csv_header = ['ParticipantId','QuestionId', 'Seq','Delta', 'Theta', 'LowAlpha', 'HighAlpha', 'LowBeta', 'HighBeta', 'LowGamma', 'MiddleGamma', 'HRV', 'Difficulty', 'Predicted']

# Merge the enhanced data and labels into a data frame
raw_feature_vector = np.column_stack((df[['ParticipantId','QuestionId', 'Seq', 'Delta', 'Theta', 'LowAlpha', 'HighAlpha', 'LowBeta', 'HighBeta', 'LowGamma', 'MiddleGamma', 'HRV', 'Difficulty']].values, y_pred_rf))
raw_df = pd.DataFrame(raw_feature_vector, columns=csv_header)

# Save the data to an output CSV file
raw_df.to_csv('./exp_2/release_data_predicted.csv', index=False)




pass