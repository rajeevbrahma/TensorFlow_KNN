# Importing Required modules

import csv # csv module
import tensorflow as tf # Tensorflow module



training_set = []
training_set_y = []

# Preparing Training Data set 
# Extracting each parameter into different list.
with open("TRAIN_SET.csv","rb") as file:
	reader = csv.reader(file)
	for row in reader:
		training_set.append([row[1],row[3],row[4],row[5]])
		training_set_y.append(row[6])

# Excluding the first column from the list (which is nothing but name column)
training_set = training_set[1:]
training_set_y = training_set_y[1:]

testing_set = []
testing_set_y = []
# Preparing Test data set
# Extracting each parameter into different list.
with open("TEST_SET.csv","rb") as file:
	reader = csv.reader(file)
	for row in reader:
		testing_set.append([row[1],row[3],row[4],row[5]])
		testing_set_y.append(row[6])

# Excluding the first column from the list (which is nothing but name column)
testing_set = testing_set[1:]
testing_set_y = testing_set_y[1:]


# Placeholder you can assign values in future its kind of a variable
#  v = ("variable type",None) -- You can assign any number of variables for v
#  v = ("variable type",4)    -- You can assign 4 variables for v
#  v = ("variable type",[None,4])  -- you can have multidimensional values here 
	# Here the no.of rows you can have any number but the columns are fixed with size 4
training_values = tf.placeholder("float",[None,len(training_set[0])])
test_values     = tf.placeholder("float",[len(training_set[0])])


# We have used the L1(Manhattan distance) formula
#  -----reduce_sum------ 

# ----'x' is [[1, 1, 1]
#          [1, 1, 1]]
# -----tf.reduce_sum(x) ==> 6
# -----tf.reduce_sum(x, 0) ==> [2, 2, 2]
# -----tf.reduce_sum(x, 1) ==> [3, 3] 
# -----tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
# -----tf.reduce_sum(x, [0, 1]) ==> 6 

# The example 3 is our case as we have only one matrix we are going to add what ever there in that matrix
distance = tf.reduce_sum(tf.abs(tf.add(training_values,tf.neg(test_values))),reduction_indices=1) 	

# tf.argmin(input,dimension,name=None)
# Returns the index with the smallest value across dimensions of a tensor
# ---dimension : A Tensor Must be one of the following types : int32, int64 . 0<=dimension<rank(input).
#    Describes which dimension of the input Tensor to reduce across. For vectors,use dimension = 0 
prediction = tf.arg_min(distance,0)


# Initializing  the session
init = tf.initialize_all_variables()


with tf.Session() as sess:
	sess.run(init)	
	# Looping through the test set to compare against the training set
	for i in range (len(testing_set)):
		# Tensor flow method to get the prediction near to the test parameters in the training set.
		index_in_trainingset = sess.run(prediction,feed_dict={training_values:training_set,test_values:testing_set[i]})	


		print "Test %d, and the prediction is %s, the real value is %s"%(i,training_set_y[index_in_trainingset],testing_set_y[i])

