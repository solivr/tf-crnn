#!/usr/bin/env python
__author__ = 'solivr'


def labelInt2Char(n):
    if 0 <= n <= 9:
		c = chr(n + 48)
    elif 10 <= n <= 35:
		c = chr(n + 97 - 10)
    elif n == 36:
		c = ''
	return c


def convertSparseArrayToStr(label_set):
	labels = []

	# Initialize results to code 36 which is '-'
	results = [[36 for _ in range(label_set[2][1])] for i in range(label_set[2][0])]  # label_set[2] = dense_shape

	for i in range(label_set[0].shape[0]):  # total number of characters
		x, y = label_set[0][i]
		results[x][y] = label_set[1][i]

	for i in range(len(results)):
		label = ''
		for j in range(len(results[i])):
			label += labelInt2Char(results[i][j])

		labels.append(label)

	return labels
