from sklearn import svm
from sklearn.svm import SVC
from a3 import *
from sklearn.svm import SVR

def svm(train_data, train_label, valid_data, valid_targets):


def svr(train_data, train_label, valid_data, valid_targets):
	clf = SVR()
	clf.fit(train_data, train_label)
	predict_labels = clf.predict(valid_data) 
	return predict_labels

if __name__ == '__main__':
	###################################################################
	# TEST #
	###################################################################

	image_data, labels, identity = loadLabeledMatFile('../a3data/labeled_images.mat')
	test_data = loadTestMatFile('../a3data/public_test_images.mat')
	image_data_scaled = preprocessing.scale(image_data)

	data_train, data_valid, label_train, label_valid = train_test_split(image_data, labels, test_size=0.33, random_state=42)
	print svr(data_train, label_train, data_valid, label_valid)