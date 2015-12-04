from sklearn.neighbors import KNeighborsClassifier

def knn(k, train_data, train_label, valid_data, valid_targets):

	neigh = KNeighborsClassifier(n_neighbors=k)
	neigh.fit(train_data, train_label)

	predict_labels = neigh.predict(valid_data)
	correctness = {}
	correctPrediction = 0
	# UNCOMMENT next line if you want to calculate correct prediction rate for training data
	# predict_labels = knn(k, train_data, train_labels, valid_data)
	# predict_labels = run_knn(k, train_inputs, train_targets, test_inputs)
	for i in range(len(predict_labels)):
		if predict_labels[i] == valid_targets[i]:
			correctPrediction += 1

	correctRate = (float(correctPrediction)/len(predict_labels)) * 100
	# correctness[k] = correctRate
	# print correctness
	return correctRate


def plotCorrectness(correctness, plot_title):
	"""
	plot keys VS values in correctness
	"""
	x = correctness.keys()
	y = correctness.values()
	plt.xlabel('k-Values')
	plt.ylabel('Correctness')
	plt.title(plot_title)
	plt.ylim(1,100)
	plt.xlim(0,11)
	plt.plot(x, y, ".")
	plt.show()