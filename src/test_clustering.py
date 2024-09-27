# Re-import necessary libraries after kernel reset

import pandas as pd
from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer

# Re-load the dataset
icd_data = pd.read_csv(DATA_PATH.joinpath("icd_grouped.csv"))

# Tokenize the icd_code_modified column
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
X = vectorizer.fit_transform(icd_data['icd_code_modified'])

# Reduce dimensionality with TruncatedSVD
svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X)

# Perform hierarchical clustering using AgglomerativeClustering
agglomerative = AgglomerativeClustering(n_clusters=None, distance_threshold=10, linkage='ward')
clusters = agglomerative.fit_predict(X_reduced)

# Add the new clusters to the dataframe
icd_data['cluster'] = clusters

# Save the updated CSV file with cluster information
output_file_path = DATA_PATH.joinpath('icd_grouped_with_clusters.csv')
icd_data.to_csv(output_file_path, index=False)