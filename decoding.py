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


def labelInt2CharWithBlank(n):
    if 0 <= n <= 9:
        c = chr(n + 48)
    elif 10 <= n <= 35:
        c = chr(n + 97 - 10)
    elif n == 36:
        c = '-'
    return c


def convertSparseArrayToStr(label_set):
    labels = []

    # Initialize results to code 36 which is '-'
    results = [[36 for _ in range(label_set[2][1])] for i in range(label_set[2][0])]  # label_set[2] = dense_shape

    for i, v in enumerate(label_set[0]):  # get all characters
        x, y = v
        results[x][y] = label_set[1][i]

    for res in results:
        label = ''
        for char_code in res:
            label += labelInt2Char(char_code)

        labels.append(label)

    return labels


def simpleDecoderWithBlank(results):
    labels = []
    for res in results:
        label = ''
        for char_code in res:
            label += labelInt2CharWithBlank(char_code)
        labels.append(label)
    return labels


def simpleDecoder(p):
    labels = simpleDecoderWithBlank(p)
    results = []
    for label in labels:
        temp_s = ''
        for i in range(len(label)):
            if label[i] != '-' and not(i > 0 and label[i] == label[i-1]):
                temp_s += label[i]
        results.append(temp_s)
    return results


def eval_accuracy(predicted, true):
    tot = len(predicted)
    equal = 0
    for i in range(tot):
        if predicted[i] == true[i]:
            equal += 1
    return float(equal) / tot

