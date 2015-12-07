import scipy.io as sio
import numpy as np
import sklearn as sklearn
import matplotlib.pyplot as plt; plt.rcdefaults()

from knn import *
from logistic_regression import *
# from pca import *
from bayes import *
from dt import *

from matplotlib import pyplot as plt
from sklearn import cross_validation
from sklearn import svm
from sklearn import datasets
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import LabelKFold
from sklearn.decomposition import RandomizedPCA

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
	# print key
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
	# print train_label.shape
	valid_data = valid_data.reshape([vd_size,1024])
	valid_label = valid_label.reshape([vd_size,])
	# print valid_label
	return train_data, valid_data, train_label, valid_label

def identityKfold(image_data, labels, identity, nfold, train_function):
	corr = []
	lkf = LabelKFold(identity, nfold)
	for train, valid in lkf:
		train_data, valid_data, train_label, valid_label = generateDataByIndex(train, valid, image_data, labels)
		predict_labels = train_function(train_data, train_label, valid_data, valid_label)
		corrRate = calculateCorr(predict_labels, valid_label)
		corr.append(corrRate)
	return corr


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99,
                        top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((w, h)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()

def pca(image_data):
	pca = RandomizedPCA(n_components=150).fit(image_data)
	transformed = pca.transform(image_data)
	# inv = pca.inverse_transform(transformed)
	return transformed_image

if __name__ == '__main__':
	
	# load data from labeled_image.mat file
	# returns a dictionary with keys 
	# ['__globals__', 'tr_labels', '__header__', 'tr_identity', '__version__', 'tr_images']
	image_data, labels, identity = loadLabeledMatFile('../a3data/labeled_images.mat')
	test_data = loadTestMatFile('../a3data/public_test_images.mat')
	image_data_scaled = preprocessing.scale(image_data)
	
	# # ramdom select data from data set as traina and validation sets
	# data_train, data_valid, label_train, label_valid = train_test_split(image_data_scaled, labels, test_size=0.33, random_state=42)

	###################################################################
	# PCA #
	###################################################################
	pca_image = pca(image_data)

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

	# corr = identityKfold(pca_image, labels, identity, 3, logistic_regression)
	# corrRate = sum(corr)/3
	# print "correctness rate for validation data:", corrRate
	# correctness rate for validation data: 65.2991452991
	# correctness rate for validation data (scaled input): 68.6837606838

	# predict_labels = logistic_regression(data_train, label_train, data_valid, label_valid)
	# c = calculateCorr(predict_labels, label_valid)
	# print "correctness rate for validation data:", c

	###################################################################
	# Bayes Classifier #
	###################################################################

	# Gaussian Naive Bayes
	# predict_labels = Gbayes(data_train, label_train, data_valid, label_valid)
	# corrR = calculateCorr(predict_labels, label_valid)
	# # corrRate = sum(corr)/3
	# print "correctness rate for validation data:", corrR
	# # correctness rate for validation data: 34.2222222222 (identitykFold)
	# # correctness rate for validation data: 36.9565217391



	###################################################################
	# Decision Tree #
	###################################################################

	# corr = identityKfold(image_data_scaled, labels, identity, 3, dt)
	# corrRate = sum(corr)/3
	# print "correctness rate for validation data:", corrRate
	# # correctness rate for validation data: 45.5486542443
	# # correctness rate for validation data: 43.1111111111 (identity k fold)

	###################################################################
	# write predict results to file #
	###################################################################

	# with open('results.csv', 'w') as csvfile:
	# 	fieldnames = ['Id', 'Prediction']
	# 	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	# 	writer.writeheader()
	# 	for i, j in zip(range(1,1254), predict_labels):
	# 		writer.writerow({'Id': i, 'Prediction': j})

	###################################################################
	# plot images #
	###################################################################
	# titles = [0]*1024

	# plot_gallery(image_data, titles, 32, 32, n_row=3, n_col=4)

