from sklearn.naive_bayes import GaussianNB
from a3 import *

def Gbayes(train_data, train_label, valid_data, valid_targets):
	gnb = GaussianNB()
	y_pred = gnb.fit(train_data, train_label).predict(valid_data)

	return y_pred