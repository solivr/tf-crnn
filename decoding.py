#!/usr/bin/env python
__author__ = 'solivr'

import tensorflow as tf

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


def levenshtein(s1, s2):  # https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def eval_WER(predicted, true):
    tot = len(predicted)
    false = 0
    for i in range(tot):
        if predicted[i] != true[i]:
            false += 1
    return float(false) / tot


def eval_CER(predicted, true):
    tot_chars = 0
    error = 0
    for w, t in zip(predicted, true):
        tot_chars += len(t)
        dist = levenshtein(w, t)
        error += dist

    return float(error) / tot_chars


def evaluation_metrics(predicted, true):
    accuracy = eval_accuracy(predicted, true)
    wer = eval_WER(predicted, true)
    cer = eval_CER(predicted, true)

    return accuracy, wer, cer