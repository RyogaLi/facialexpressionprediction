import numpy as np
from a3 import *
from sklearn import mixture

def gmm(train_data, train_label, valid_data, valid_targets):
	# np.random.seed(1)
	print train_data.shape
	c_type = ['spherical','tied','diag','full']
	for i in c_type:
		g = mixture.GMM(n_components=7,n_iter=2000,covariance_type=i)
		print g.converged_
		t = g.fit(train_data,train_label)
		print t
		predict_label = g.predict(valid_data)
		print predict_label

if __name__ == '__main__':

	###################################################################
	# TEST GMM #
	###################################################################

	image_data, labels, identity = loadLabeledMatFile('../a3data/labeled_images.mat')
	test_data = loadTestMatFile('../a3data/public_test_images.mat')
	image_data_scaled = preprocessing.scale(image_data)

	data_train, data_valid, label_train, label_valid = train_test_split(image_data, labels, test_size=0.33, random_state=42)
	gmm(data_train, label_train, data_valid, label_valid)
