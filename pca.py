import numpy as np
from sklearn.decomposition import PCA

def pca(image_data):
	pca = PCA(n_components=2)
	transformed = pca.transform(image_data)
	# inv = pca.inverse_transform(transformed)
	print transformed