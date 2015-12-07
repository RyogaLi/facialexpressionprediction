from sklearn.neighbors import KNeighborsClassifier

def knn(k, train_data, train_label, valid_data, valid_targets):

	neigh = KNeighborsClassifier(n_neighbors=k)
	neigh.fit(train_data, train_label)

	predict_labels = neigh.predict(valid_data)
	
	return predict_labels

def calculateCorr(predict_labels, valid_targets):
	correctness = {}
	correctPrediction = 0

	for i in range(len(predict_labels)):
		if predict_labels[i] == valid_targets[i]:
			correctPrediction += 1

	correctRate = (float(correctPrediction)/len(predict_labels)) * 100
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