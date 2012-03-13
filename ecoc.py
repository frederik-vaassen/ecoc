#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#	  construct_codes.py
#
#	  Frederik Vaassen <frederik.vaassen@ua.ac.be>
#	  Copyright 2011 CLiPS Research Center
#
#	  This program is free software; you can redistribute it and/or modify
#	  it under the terms of the GNU General Public License as published by
#	  the Free Software Foundation; either version 2 of the License, or
#	  (at your option) any later version.
#
#	  This program is distributed in the hope that it will be useful,
#	  but WITHOUT ANY WARRANTY; without even the implied warranty of
#	  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	  GNU General Public License for more details.
#
#	  You should have received a copy of the GNU General Public License
#	  along with this program; if not, write to the Free Software
#	  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#	  MA 02110-1301, USA.
#

had4 = [	[1,1,1],
			[0,1,0],
			[1,0,0],
			[0,0,1]
		]

had8 =  [   [1,1,1,1,1,1,1],
			[0,1,0,1,0,1,0],
			[1,0,0,1,1,0,0],
			[0,0,1,1,0,0,1],
			[1,1,1,0,0,0,0],
			[0,1,0,0,1,0,1],
			[1,0,0,0,0,1,1],
			[0,0,1,0,1,1,0]
		]

had12 = [   [0,0,0,0,0,0,0,0,0,0,0],
			[1,0,1,0,0,0,1,1,1,0,1],
			[1,1,0,1,0,0,0,1,1,1,0],
			[0,1,1,0,1,0,0,0,1,1,1],
			[1,0,1,1,0,1,0,0,0,1,1],
			[1,1,0,1,1,0,1,0,0,0,1],
			[1,1,1,0,1,1,0,1,0,0,0],
			[0,1,1,1,0,1,1,0,1,0,0],
			[0,0,1,1,1,0,1,1,0,1,0],
			[0,0,0,1,1,1,0,1,1,0,1],
			[1,0,0,0,1,1,1,0,1,1,0],
			[0,1,0,0,0,1,1,1,0,1,1]
		]

had16 = [   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,1,0,1,1,1,0,0,0,1,0,1,1,1],
			[1,0,0,1,0,1,1,0,1,0,0,1,0,1,1],
			[1,1,0,0,1,0,1,0,1,1,0,0,1,0,1],
			[1,1,1,0,0,1,0,0,1,1,1,0,0,1,0],
			[0,1,1,1,0,0,1,0,0,1,1,1,0,0,1],
			[1,0,1,1,1,0,0,0,1,0,1,1,1,0,0],
			[0,1,0,1,1,1,0,0,0,1,0,1,1,1,0],
			[0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
			[0,1,1,0,0,1,1,1,1,0,0,1,1,0,0],
			[1,0,1,0,1,0,1,1,0,1,0,1,0,1,0],
			[1,1,0,0,1,1,0,1,0,0,1,1,0,0,1],
			[0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
			[0,1,1,1,1,0,0,1,1,0,0,0,0,1,1],
			[1,0,1,1,0,1,0,1,0,1,0,0,1,0,1],
			[1,1,0,1,0,0,1,1,0,0,1,0,1,1,0]
		]

had24 =	[	[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1],
			[1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1],
			[1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1],
			[1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1],
			[1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0],
			[0,1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1],
			[1,0,1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0],
			[0,1,0,1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,1],
			[1,0,1,0,1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1],
			[1,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0],
			[0,1,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0],
			[0,0,1,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1],
			[1,0,0,1,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0,1,0,0,1],
			[1,1,0,0,1,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0,1,0,0],
			[0,1,1,0,0,1,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0,1,0],
			[0,0,1,1,0,0,1,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0,1],
			[1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0],
			[0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1,1,0,0,0,0,1],
			[1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1,1,0,0,0,0],
			[0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1,1,0,0,0],
			[0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1,1,0,0],
			[0,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1,1,0],
			[0,0,0,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1,1]
		]

import re
import os
import sys
import math
import time
import operator
from random import shuffle
from optparse import OptionParser
from itertools import combinations

liblinearpath = '/home/frederik/Tools/liblinear-1.8'
libsvmpath = '/home/frederik/Tools/libsvm-3.11'

def calculate_separation(matrix):
	'''
	Calculates the minimum Hamming distances between rows and columns in a
	matrix and calculates its error-correcting capacity.

	'''
	if isinstance(matrix, dict):
		keys = []
		values = []
		for key, value in matrix.items():
			keys.append(key)
			values.append(value)
		matrix = values

	min_row = 9999
	for index, row1 in enumerate(matrix):
		tempmatrix = matrix[:]
		del(tempmatrix[index])
		for index2, row2 in enumerate(tempmatrix):
			d = sum(e1 != e2 for e1, e2 in zip(row1, row2))
			if d < min_row:
				min_row = d
	hd = int((min_row - 1)/2)
	if hd < 0:
		hd = 0

	min_col = 9999
	for index, col1 in enumerate(zip(*matrix)):
		tempmatrix = zip(*matrix)
		del(tempmatrix[index])
		for col2 in tempmatrix:
			d = sum(e1 != e2 for e1, e2 in zip(col1, col2))
			if d < min_col:
				min_col = d

	print 'Minimum row Hamming distance: {0}.'.format(min_row)
	print 'Minimum column Hamming distance: {0}.'.format(min_col)
	print 'These output codes (of length {2}) will correct {0} incorrect bits ({1:2.2f}% of the total amount of classifiers).\n'.format(hd, 100*hd/float(len(matrix[0])), len(matrix[0]))

def onevsall(label_map):
	'''
	Sets up the matrix and subtasks in a one-vs-all configuration.

	'''
	classes = label_map.keys()
	matrix = {}
	for i, label in enumerate(classes):
		matrix[label] = [int(i==j) for j in range(len(classes))]

	temp_matrix = dict([(label_map[key], value) for key, value in matrix.items()])
	tasks = getSubtasks(temp_matrix)

	return matrix, tasks

def hadamard(label_map):
	'''
	Takes a list of classes as input and returns a matrix of output codes based on
	Hadamard matrices and a series of data subdivisions (tasks). Hadamard matrices
	only exist in certain dimensions (4, 8, 12...). Should the number of classes not
	match any matrix, the last rows will be pruned from the first matrix large
	enough to accomodate the number of classes.

	'''
	classes = label_map.keys()
	matrix = {}

	if len(classes) < 3:
		sys.exit('Using ECOCs for less than three classes is pointless. Exiting.')
	if 3 < len(classes) <= 4:
		size = 3
		for i, label in enumerate(classes):
			matrix[label] = had4[i]
	elif 5 < len(classes) <= 8:
		size = 7
		for i, label in enumerate(classes):
			matrix[label] = had8[i]
	elif 9 < len(classes) <= 12:
		size = 11
		for i, label in enumerate(classes):
			matrix[label] = had12[i]
	elif 13 < len(classes) <= 16:
		size = 15
		for i, label in enumerate(classes):
			matrix[label] = had16[i]
	elif 17 < len(classes) <= 24:
		size = 23
		for i, label in enumerate(classes):
			matrix[label] = had24[i]
	elif 180 < len(classes) <= 188:
		size = 187
		for i, label in enumerate(classes):
			matrix[label] = had188[i]
	else:
		print len(classes)
		raise NotImplementedError('Hadamard matrices larger than x24 have not been implemented yet. See http://www2.research.att.com/~njas/hadamard/ to find larger matrices.')

	temp_matrix = dict([(label_map[key], value) for key, value in matrix.items()])
	tasks = getSubtasks(temp_matrix)

	return matrix, tasks

def dietterich(label_map):
	'''
	Set up a codewords matrix after Dietterich95. Warning, creates long codewords!

	'''
	classes = label_map.keys()
	k = len(classes)

	if k < 3:
		sys.exit('There are no more than two classes, making the use of ECOC unneccesary. Exiting.')
	elif 3 <= k <= 11:
		size = 2**(k-1)-1
		matrix = {}
		matrix[classes[0]] = [1]*size
		matrix[classes[1]] = [0]*(2**(k-2)) + [1]*(2**(k-2)-1)
		matrix[classes[2]] = [0]*(2**(k-3)) + [1]*(2**(k-3)) + [0]*(2**(k-3)) + [1]*(2**(k-3)-1)
		for i in range(3, k):
			matrix[classes[i]] = [0]*(2**(k-i-1))
			while len(matrix[classes[i]]) < size:
				matrix[classes[i]] += [1]*(2**(k-i-1))
				matrix[classes[i]] += [0]*(2**(k-i-1))
			matrix[classes[i]] = matrix[classes[i]][:size]

		temp_matrix = dict([(label_map[key], value) for key, value in matrix.items()])
		tasks = getSubtasks(temp_matrix, size)

	else:
		raise NotImplementedError('ECOC for more than 11 classes has not been implemented yet.')

	return matrix, tasks

def getSubtasks(matrix, size=None):
	'''
	Turns the row-based structure of ECOC matrices to a column-based structure
	containing all the data subdivisions that need to be trained with.

	'''
	if not size:
		size = len(matrix.values()[0])

	tasks = []
	for i in range(size):
		task = {}
		for label in matrix.keys():
			TorF = bool(matrix[label][i])
			if TorF:
				task[label] = '+1'
			else:
				task[label] = '-1'
		tasks.append(task)

	return tasks

def trainTasks(train_file, tasks, fold_num, output_folder=None, params={'-c': 1}):
	'''
	Given a training instance file and a list of tasks, adapt the training
	vectors to fit the tasks and build an SVM model for each of them.

	'''
	if not output_folder:
		output_folder = 'models'
	output_folder = os.path.join(output_folder, 'fold-{0:02d}'.format(fold_num+1))
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	model_files = []
	labels, instances = read_problem(train_file)
	for i, task in enumerate(tasks):
		print '---training task {0:03d}/{1:03d}'.format(i+1,len(tasks))
		new_labels = [int(task[label]) for label in labels]

		paramstring = ''
		for param, value in params.items():
			paramstring += ' {0} {1}'.format(param, value)
		paramstring += ' -q'

		model_file = os.path.join(output_folder, os.path.basename(train_file) + '.task{0:03d}.model'.format(i+1))
		model = train(new_labels, instances, paramstring)
		save_model(model_file, model)

		model_files.append(model_file)

	return model_files

def normalizePrediction(value):
	'''
	Normalize the decision value to a value between 0 and 1.

	'''
	value = math.atan(value)/math.pi + 0.5
	return value

def getPredictions(test_file, tasks, model_files):
	'''
	Returns probability values of the +1 class per task as well as task
	accuracies (classification accuracies per task).

	Requires a test instance file, a list of tasks, and corresponding model
	files. Transform the vectors to fit the tasks and classifies them against
	the matching SVM model.

	'''
	assert len(tasks) == len(model_files), 'Not as many model files as tasks'

	labels, instances = read_problem(test_file)

	models = []
	print '---Loading models...'
	for model_file in model_files:
		models.append(load_model(model_file))
	print '---Done.'

	task_accuracies = [0.0 for _ in range(len(tasks))]

	predictions = []
	for label, instance in zip(labels, instances):
		instance_predictions = []
		for i, (model, task) in enumerate(zip(models, tasks)):
			new_label = int(task[label])

			pred_labels, ACC, pred_values, label_order = predict([new_label], [instance], model)
			assert len(pred_values[0]) == 1
			pred_value = pred_values[0][0]

			# If the label order is reversed, reverse the sign of the distance value.
			if label_order == [-1, 1]:
				pred_value = -pred_value
			# Normalize the value and add it to the instance predictions.
			instance_predictions.append(normalizePrediction(pred_value))

			# Add one if the prediction was accurate
			if new_label == pred_labels[0]:
				task_accuracies[i] += 1

		predictions.append(instance_predictions)

	task_accuracies = [score/len(instances) for score in task_accuracies]

	return predictions, task_accuracies

def hamming(list1, list2):
	'''
	Calculate the Hamming distance between two lists.

	'''
	return sum([i != j for i,j in zip(list1, list2)])

def euclidian(list1, list2):
	'''
	Calculate the Euclidian distance between two lists.

	'''
	return math.sqrt(sum((i - j)**2 for i, j in zip(list1, list2)))

def decode(prediction, codewords, mode=euclidian):
	'''
	Decode a prediction against a dictionary of codewords.

	'''
	distances = sorted([(label, mode(prediction, codeword)) for label, codeword in codewords.items()], key=operator.itemgetter(1))
	at_minimum = [label for (label, distance) in distances if distance == distances[0][1]]

	return at_minimum

def getLabelMap(folder):
	'''
	Retrieves the label map from the first svm.metadata.txt in <folder>.

	'''
	for root, dirs, files in os.walk(os.path.abspath(folder)):
		if 'svm.metadata.txt' in files:
			with open(os.path.join(root, 'svm.metadata.txt'), 'r') as fin:
				lines = [line.strip().split(' > ') for line in fin.readlines()]
				mapping = {}
				for line in lines:
					mapping[line[1].split('Class_')[1]] = int(line[0])
				return mapping

def getInstances(folder, pattern=None):
	'''
	Returns a list of tuples containing the instance files for (train, test) per
	fold.
	Loops through <folder> and return all SVM instance files matching the
	specified filters. If no filter is specified, try to find the unfiltered
	token unigram TotalSet files. Falls back to the first instance file in
	alphabetical order.

	Output:
	[(fold-01.train, fold-01.test), (fold-02.train, fold-02.test),...]

	'''
	folds = sorted([os.path.join(os.path.abspath(folder), f) for f in os.listdir(folder) if re.match('fold-\d+', f)])

	if pattern:
		pattern = re.compile(pattern)
	else:
		pattern = re.compile('N_GRAM_TOKEN_?1.TotalSet.ngrams.txt')

	if len(folds) == 1:
		for root, dirs, files in os.walk(folds[0]):
			if root.endswith('train/svm'):
				instances = sorted([os.path.join(root, f) for f in files if re.search(pattern, f)])
		instances = [(instances[0], None)]
	else:
		instances = []
		for fold in folds:
			fold_inst = [None, None]
			for root, dirs, files in os.walk(fold):
				path = os.path.split(root)
				if path[1] == 'svm':
					path = os.path.split(path[0])
					if path[1] == 'train':
						fold_inst[0] = sorted([os.path.join(root, f) for f in files if re.search(pattern, f)])[0]
					elif path[1] == 'test':
						fold_inst[1] = (sorted([os.path.join(root, f) for f in files if re.search(pattern, f)])[0] or None)
			assert fold_inst[0] is not None
			assert fold_inst[1] is not None
			instances.append(tuple(fold_inst))

	return instances

def evaluate(predictions, codewords, label_map, output_file):
	'''
	Given a list of predictions by all dichotomizers, decode the label using the
	codewords and write the decoded labels followed by the predictions to
	output_file.

	'''
	if not os.path.exists(os.path.split(output_file)[0]):
		os.makedirs(os.path.split(output_file)[0])

	pred_labels = []
	with open(output_file, 'w') as fout:
		for i, prediction in enumerate(predictions):
			labels = decode(prediction, codewords, euclidian)
			if len(labels) > 1:
				print '---Warning, multiple possible labels for instance {0}: {1}'.format(i, ' '.join(labels))
				shuffle(labels)
			pred_labels.append(labels[0])
			fout.write('{0}\t{1}\n'.format(label_map[labels[0]], '\t'.join(map(str, prediction))))
		print 'Predictions written to {0}.'.format(output_file)
	return pred_labels

def getModelFiles(folder):
	'''
	Returns a list of lists of model files. Use this if you don't need to
	re-train but want to use existing model files. Returns an empty list if no
	model files were found.

	'''
	models = []
	if os.path.exists(folder):
		folds = sorted([os.path.join(folder, f) for f in os.listdir(folder) if re.match('fold-\d+', f)])
		for fold in folds:
			fold_models = sorted([os.path.join(fold, f) for f in os.listdir(fold) if os.path.splitext(f)[1] == '.model'])
			if fold_models:
				models.append(fold_models)
			else:
				models = []
				break

	return models

def correlate(test_file, matrix, predictions_file):
	'''
	Given a test file, a codeword matrix and a file with dichotomizer predictions,
	see which dichotomizer was right or wrong where. Then calculate the correlation
	between dichotomizers according to Qav (see Kuncheva and Whitaker 2003).

	'''
	with open(test_file, 'r') as fin:
		correct = [matrix[line.strip().split('\t')[-1]] for line in fin.readlines() if line.strip()]
	with open(predictions_file, 'r') as fin:
		probabilities = [map(float, line.strip().split('\t')) for line in fin.readlines() if line.strip()]
	assert len(correct) == len(probabilities)

	results = [[] for _ in range(len(probabilities[0]))]
	for probs, gold in zip(probabilities, correct):
		for i, (prob, corr) in enumerate(zip(probs, gold)):
			# Check if the probability matches the codeword bit.
			if round(prob) == corr:
				results[i].append(1)
			else:
				results[i].append(0)

	# Calculate average Q correlation.
	q_vals = []
	for i, j in combinations(range(len(results)),2):
		assert len(results[i]) == len(results[j])
		zipped = zip(results[i], results[j])
		n_both_correct = float(len([None for (d1, d2) in zipped if d1 == d2 == 1]))
		n_both_incorrect = float(len([None for (d1, d2) in zipped if d1 == d2 == 0]))
		n_i_correct = float(len([None for (d1, d2) in zipped if (d1 == 1 and d2 == 0)]))
		n_j_correct = float(len([None for (d1, d2) in zipped if (d1 == 0 and d2 == 1)]))
		#~ print '11:', n_both_correct
		#~ print '00:', n_both_incorrect
		#~ print '10:', n_i_correct
		#~ print '01:', n_j_correct
		assert n_both_correct + n_both_incorrect + n_i_correct + n_j_correct == len(results[i])
		q = (n_both_correct * n_both_incorrect - n_j_correct * n_i_correct) / (n_both_correct * n_both_incorrect + n_j_correct * n_i_correct)
		print 'Q({0},{1}): {2}'.format(i, j, q)
		q_vals.append(q)
	q_av = (2.0 / (float(len(results)) * (float(len(results)) - 1))) * sum(q_vals)
	print 'Average Q:', q_av

	return q_av

def main(instance_folder, results_folder, ecoc_type='dietterich', pattern=None, params={'-c': 1}, reuse_models=False):
	'''
	Returns a list of lists of predicted labels (per fold).

	Contains the main pipeline:

	* Generates ECOC matrix and subtasks based on the
	ECOC type (default the construction method from Dietterich95).

	* Trains the dichotomizers based on the instances in instance_folder. By
	default the script looks for files with the following filenames:
	N_GRAM_TOKEN_?1.TotalSet.ngrams.txt or
	Use the <pattern> option to specify your own pattern (can be a regex).

	* Tests each instance against each dichotomizer, decodes the resulting
	codeword, and writes the results to files in results_folder. Returns the
	(human-readable) labels as a list of lists (per fold).

	'''
	instance_files = getInstances(instance_folder, pattern=pattern)
	labels = getLabelMap(instance_folder)

	# Prepare the ECOC matrices and subtasks.
	if ecoc_type.lower() == 'dietterich':
		matrix, tasks = dietterich(labels)
	elif ecoc_type.lower() == 'hadamard':
		matrix, tasks = hadamard(labels)
	elif ecoc_type.lower() == 'onevsall':
		matrix, tasks = onevsall(labels)
	else:
		sys.exit('Unknown ECOC type: {0}\n Allowed values: dietterich, hadamard, onevsall.'.format(ecoc_type))
	for label, code in matrix.items():
		print '{0:10}\t{1}'.format(label, code)
	calculate_separation(matrix)

	# Gather existing model files.
	model_files = getModelFiles(results_folder)

	# Remove old model files if requested.
	if not reuse_models and model_files:
		for fold in model_files:
			for model_file in fold:
				os.remove(model_file)
		model_files = []

	if not model_files:
		if reuse_models:
			print 'WARNING: No existing model files found! Re-generating models.'
			while True:
				x = raw_input('Enter "q" to quit or any other key to continue.\n>>> ')
				if x == 'q':
					sys.exit('Script interrupted.')
				else:
					break
		model_files = []
		# Train classifiers for each fold/task.
		for i, (train_file, test_file) in enumerate(instance_files):
			print 'Training fold {0}/{1}'.format(i+1, len(instance_files))
			models = trainTasks(train_file, tasks, i, output_folder=results_folder, params=params)
			model_files.append(models)
	else:
		print 'Using existing models.'

	predicted_labels = []
	# Test each task for each fold.
	for i, (instance_file_pair, fold_models) in enumerate(zip(instance_files, model_files)):

		train_file, test_file = instance_file_pair

		# Classify!
		print 'Classifying fold {0}/{1}'.format(i+1, len(instance_files))
		predictions, task_accuracies = getPredictions(test_file, tasks, fold_models)

		# Decode the predictions into the actual class labels.
		output_file = os.path.join(results_folder, 'fold-{0:02d}'.format(i+1), os.path.basename(test_file) + '.predictions')
		fold_predicted_labels = evaluate(predictions, matrix, labels, output_file)
		predicted_labels.append(fold_predicted_labels)

	return predicted_labels

if __name__ == '__main__':
	parser = OptionParser(usage = '''
python %prog instance_folder (options)

Takes a folder of Stylene train and test instance files (as structured by
styleneFolding.py) as input. Generates and trains ECOC dichotomizers, tests
each instance against the dichotomizers and decodes the predictions. Writes
the resulting labels and probabilities to output_folder/fold-XX/*.predictions.''', version='%prog 0.1')
	parser.add_option('-i', '--instance-pattern', dest='pattern', default=None,
						help='Specify the file name of the instance files you want to use, can be a regex pattern. (Default: N_GRAM_TOKEN_?1.TotalSet.ngrams.txt)')
	parser.add_option('-o', '--output-folder', dest='output_folder', default=None,
						help="Specify the folder you want model files and predictions to be stored. (Default: instance_folder/fold-XX/)")
	parser.add_option('-m', '--coding-matrix', dest='ecoc_type', default='dietterich',
						help="Specify the type of ECOC matrix you want to use. (Available: 'dietterich' (default), 'hadamard', 'onevsall')")
	parser.add_option('-c', '--classifier', dest='classifier', default='libsvm',
						help="Specify the classifier you want to use. (Available: 'libsvm' (default), 'liblinear')")
	parser.add_option('-p', '--params', dest='params', default='c=1',
						help="Specify the classifier parameters you want to use. Format: c=1/g=1/... (Default: c=1)")
	parser.add_option('-f', '--force-reuse', dest='reuse_models', default=False, action='store_true',
						help="Use this option to force the re-use of model files if they already exist. It is YOUR responsibility to make sure these model files match your test instances!")
	(options, args) = parser.parse_args()

	if len(args) != 1:
		sys.exit(parser.print_help())

	input_folder = args[0]
	if options.output_folder is None:
		output_folder = input_folder
	else:
		output_folder = options.output_folder

	params = dict([(('-'+param.lstrip('-')).split('=')) for param in options.params.split('/')])

	if options.classifier == 'libsvm':
		sys.path.insert(0, os.path.join(libsvmpath, 'python'))
		from svmutil import svm_read_problem as read_problem
		from svmutil import svm_train as train
		from svmutil import svm_save_model as save_model
		from svmutil import svm_predict as predict
	elif options.classifier == 'liblinear':
		sys.path.insert(0, os.path.join(liblinearpath, 'python'))
		from liblinearutil import read_problem
		from liblinearutil import train
		from liblinearutil import save_model
		from liblinearutil import predict

	main(input_folder, output_folder, ecoc_type=options.ecoc_type, pattern=options.pattern, params=params, reuse_models=options.reuse_models)
