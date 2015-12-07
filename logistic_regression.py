import numpy as np
import matplotlib.pyplot as plt
import csv
from a3 import *
from sklearn import linear_model, datasets

def logistic_regression(train_data, train_label, valid_data, valid_targets):
	
	h = 0.2
	logreg = linear_model.LogisticRegression()
	# print train_data.shape
	# print train_label.shape
	logreg.fit(train_data, train_label)

	predict_labels = logreg.predict(valid_data)
	log = logreg.predict_log_proba(valid_data)
	# print log
	return predict_labels

def plotLogProba():
	pass


if __name__ == '__main__':
	
	## TEST LOG_REG ###
	image_data, labels, identity = loadLabeledMatFile('../a3data/labeled_images.mat')
	test_data = loadTestMatFile('../a3data/public_test_images.mat')
	data_train, data_valid, label_train, label_valid = train_test_split(image_data, labels, test_size=0.33, random_state=42)

	predict_labels = logistic_regression(data_train, label_train, data_valid, label_valid)

