import scipy.io as sio
import numpy as np
import sklearn as sklearn
import matplotlib.pyplot as plt; plt.rcdefaults()
from knn import *
from logistic_regression import *
from sklearn import cross_validation
from sklearn import svm
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import LabelKFold

def loadTestMatFile(filename):
	mat_contents = sio.loadmat(filename)
	data = mat_contents['public_test_images'].T
	num_images = mat_contents['public_test_images'].shape[2]
	image_data = np.reshape(data, (num_images, 1024))
	return image_data

def loadLabeledMatFile(filename):
	"""
	load labeled.mat file and return transformed image_data and associated labels
	"""
	mat_contents = sio.loadmat(filename)
	key = mat_contents.keys()
	print key
	data = mat_contents['tr_images'].T
	num_images = mat_contents['tr_images'].shape[2]
	identity = mat_contents['tr_identity']
	image_data = np.reshape(data, (num_images, 1024))
	labels = mat_contents['tr_labels'].T[0]

	return image_data, labels, identity


def generateDataByIndex(train_index, valid_index, image_data, labels):
	"""
	Generate data sets by index

	train_index: np array 

	"""
	train_data = np.array([])
	valid_data = np.array([])
	train_label = np.array([])
	valid_label = np.array([])
	td_size = len(train_index)
	vd_size = len(valid_index)

	for index in range(image_data.shape[0]):
		if index in train_index:
			train_data = np.append(train_data, image_data[index])
			train_label = np.append(train_label, labels[index])
		elif index in valid_index:
			valid_data = np.append(valid_data, image_data[index])
			valid_label = np.append(valid_label, labels[index])	
		# print index
	# print td_size
	# print vd_size

	train_data = train_data.reshape([td_size,1024])
	train_label = train_label.reshape([td_size,])
	print train_label.shape
	valid_data = valid_data.reshape([vd_size,1024])
	valid_label = valid_label.reshape([vd_size,])
	# print valid_label
	return train_data, valid_data, train_label, valid_label

def identityKfold(image_data, labels, identity, nfold, train_function):
	corr = []
	lkf = LabelKFold(identity, nfold)
	for train, valid in lkf:
		train_data, valid_data, train_label, valid_label = generateDataByIndex(train, valid, image_data, labels)
		corr.append(train_function(train_data, train_label, valid_data, valid_label))
	return corr

if __name__ == '__main__':
	
	# load data from .mat file
	# returns a dictionary with keys 
	# ['__globals__', 'tr_labels', '__header__', 'tr_identity', '__version__', 'tr_images']
	image_data, labels, identity = loadMatFile('../a3data/labeled_images.mat')

	# # ramdom select data from data set as traina and validation sets
	# data_train, data_valid, label_train, label_valid = train_test_split(image_data, labels, test_size=0.33, random_state=42)

	###################################################################
	# knn with identity-k-fold #
	###################################################################
	# kValue = [2,4,6,8,10]
	# c = {}
	# for k in kValue:
	# 	corr = identityKfold(identity, 3, knn)
	# 	c[k] = sum(corr) / 3
	# 	corr = []
	# print "identity-k-fold with knn:", c

	# # n_fold = 3
	# plot_title = 'identity-k-fold with knn (n_fold = 3)'
	# # c = {8: 52.478632478632484, 2: 51.28205128205128, 4: 53.53846153846154, 10: 52.78632478632479, 6: 53.641025641025635}
	# plotCorrectness(c, plot_title)

	###################################################################
	# LOG_REG with identity-k-fold #
	###################################################################

	corr = identityKfold(image_data, labels, identity, 3, logistic_regression)
	corrRate = sum(corr)/3
	print corrRate



