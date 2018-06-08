import numpy as np
import pdb

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def preProcess(csv_file):
	num_features = 6
	data = np.empty((0,num_features))
	max_age = np.nanmax(csv_file['age'])
	ave_age = np.nanmean(csv_file['age']) / max_age
	max_fair = np.nanmax(csv_file['fair'])
	max_siblings= np.nanmax(csv_file['sibsp'])
	max_parch = np.nanmax(csv_file['parch'])
	labels = np.array([])
	for passenger in csv_file:
		survived = passenger[1]	
		pclass = passenger[2]
		male = 1.0 if passenger[5] == 'male' else 0.0
		age = ave_age if np.isnan(passenger[6]) else passenger[6] / max_age
		siblings = passenger[7] / max_siblings
		parch = passenger[8] / max_parch
		fair = passenger[10] / max_fair
		labels = np.append(labels, survived)
		data = np.append(data, [[pclass, male, age, siblings, parch, fair]], axis=0)
	return (data, labels)


def getData():
	train_csv = np.genfromtxt('train.csv', delimiter=',', skip_header=1, dtype=None, names = \
	"id, survived, pclass, name1, name2, sex, age, sibsp, parch, ticket, fair, cabin, embark")
	train_data, train_labels = preProcess(train_csv)
	return preProcess(train_csv)

class Network:
	def __init__(self, hidden_units, dataset):
		self.train_data, self.train_labels, self.test_data, self.test_labels = dataset				        
		num_features = self.test_data.shape[1]  
		bound = 0.05
		self.output_weights = np.random.uniform(low=-bound, high=bound, size=(1, hidden_units + 1))          # init w & b for output layer
		self.hidden_weights = np.random.uniform(low=-bound, high=bound, size=(hidden_units, num_features + 1))          # init w & b for hidden layer
		self.output_momentum = np.zeros_like(self.output_weights, dtype=np.float64)												  # init momentum for output layer
		self.hidden_momentum = np.zeros_like(self.hidden_weights, dtype=np.float64)													# init momentum for hidden layer

		# append 1 for the bias
		self.train_data = np.append(self.train_data, np.ones((self.train_data.shape[0], 1), dtype=np.float64), 1)
		self.test_data = np.append(self.test_data, np.ones((self.test_data.shape[0], 1), dtype=np.float64), 1)

	
	# Train the network using stochastic gradient decent
	def train(self, epochs=50, batch_size=100, step=0.1, momentum=0.9):
		# partition labels and data into batches for SGD
		data_batches = [self.train_data[k:k+batch_size] for k in range(0,self.train_data.shape[0], batch_size)] 
		labels = [0.9 if label == 1.0 else 0.1 for label in self.train_labels]
		labels = np.array(labels)
		label_batches = [labels[k:k+batch_size] for k in range(0,len(labels), batch_size)]										  

		self.printAccuracy(0)
		# train for epoch #
		for epoch in range(1, epochs + 1):
			# train for batch #
			for batch in range(len(data_batches)):
				self.sgd(step, momentum, data_batches[batch], label_batches[batch])
			self.printAccuracy(epoch)
			#pdb.set_trace()

		print self.confusionMatrix()
		print "DONE"
		

	# perform stochastic gradient decent (sgd) with batches 
	def sgd(self, step, momentum, data, labels):
		# feed through entire batch of data through the network
		hidden_activations, output_activations = self.feedThrough(data)

		# computer output error and then backprogapate it to compute the error at the hidden layer
		# self.output_weights[:,:-1] is needed to slice of the biases in the weights array for the output layer
		# this is done for every image in the batch
		output_error = (output_activations - labels.transpose()) * (1 - output_activations) * output_activations
		hidden_error = (np.dot(self.output_weights[:,:-1].transpose(), output_error)) * (1 - hidden_activations) * (hidden_activations)

		# compute delta_w for every weight in output and hidden layer for every image
		delta_output_weights = [np.tensordot(output_error[:,k], np.append(hidden_activations[:,k].transpose(), 1), axes=0) for k in range(output_error.shape[1])]
		delta_hidden_weights = [np.tensordot(hidden_error[:,k], data[k,:], axes=0) for k in range(hidden_error.shape[1])]

		# delta_w over every image and multiply by step size and add momemtum 
		delta_output_weights = (step / data.shape[0]) * np.sum(delta_output_weights, axis=0) + (momentum * self.output_momentum)
		delta_hidden_weights = (step / data.shape[0]) * np.sum(delta_hidden_weights, axis=0) + (momentum * self.hidden_momentum)

		# perform gradient decent
		self.output_weights -= delta_output_weights
		self.hidden_weights -= delta_hidden_weights

		# set new momentum
		self.output_momentum = delta_output_weights
		self.hidden_momentum = delta_hidden_weights

	
	# Compute the accuracy of the network on a batch
	# of data and labels
	def Accuracy(self, data, labels):
		h, output = self.feedThrough(data)
 		# take index of the neruon with the max value as the network output for an image
		#pdb.set_trace()
		output = output.flatten()
		output[output > 0.5] = 1.0
		output[output <= 0.5] = 0.0
		return 100.0 * np.sum(output == labels) / float(output.shape[0])

	
	# feed through a batch of data and return 
	# the activations for each image at the hidden 
	# layer and the output layer
	def feedThrough(self, data):
		hidden_activations = sigmoid(np.dot(self.hidden_weights, data.transpose()))
		# remember to append 1 to the activation of the hidden layer to account for the bias neuron
		output_activations = sigmoid(np.dot(self.output_weights, np.append(hidden_activations, np.ones((1, hidden_activations.shape[1])), 0)))
		return (hidden_activations, output_activations) 

	# Computer the confusion matrix on the test data
	def confusionMatrix(self):
		h, output = self.feedThrough(self.test_data) # ignore the hidden activations
		output = output.flatten()
		output[output > 0.5] = 1.0
		output[output <= 0.5] = 0.0
		C_matrix = np.zeros((2,2), dtype=np.int32)
		np.add.at(C_matrix, (self.test_labels.astype(int), output.astype(int)), 1)
		return C_matrix

	# helper function used to print the
	# training and test accuracy agains epoch number
	def printAccuracy(self, epoch_number):
		print str(epoch_number) + "," + \
					str(self.Accuracy(self.train_data, self.train_labels)) + "," + \
					str(self.Accuracy(self.test_data, self.test_labels))

data, labels = getData()
data_labels = zip(data, labels)
np.random.shuffle(data_labels)
data, labels = zip(*data_labels)
data = np.array(data)
labels = np.array(labels)
split = int(0.7 * data.shape[0])
train_d = data[:split]
train_l = labels[:split]
test_d = data[split:]
test_l = labels[split:]

Network(2, (train_d, train_l, test_d, test_l)).train(batch_size = 10, step = 0.08, epochs=50, momentum=0.9)
