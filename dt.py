from a3 import *
from sklearn import tree

def dt(train_data, train_label, valid_data, valid_targets):
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(train_data, train_label)
	predict_label = clf.predict(valid_data)

	return predict_label


# if __name__ == '__main__':
# 	###################################################################
# 	# TEST DT #
# 	###################################################################

# 	image_data, labels, identity = loadLabeledMatFile('../a3data/labeled_images.mat')
# 	test_data = loadTestMatFile('../a3data/public_test_images.mat')
# 	image_data_scaled = preprocessing.scale(image_data)

# 	data_train, data_valid, label_train, label_valid = train_test_split(image_data, labels, test_size=0.33, random_state=42)
# 	dt(data_train, label_train, data_valid, label_valid)