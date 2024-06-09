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

############################################################################################
When the paper is published, the corresponding code will be released.
############################################################################################

pass
