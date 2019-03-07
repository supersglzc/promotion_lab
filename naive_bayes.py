from math import *


def build(train):
    total = [0, 0]
    attributes = []
    probability = []
    m = len(train[0])
    for i in range(m):
        attributes.append({})
        probability.append({})
    for row in train:
        for i in range(m):
            if row[i] not in attributes[i]:
                attributes[i][row[i]] = [0, 0]
                probability[i][row[i]] = [0, 0]
            attributes[i][row[i]][int(row[-1])] += 1
        total[int(row[-1])] += 1
    for i in range(m):
        for (j, k) in attributes[i].items():
            probability[i][j][0] = k[0] / total[0]
            probability[i][j][1] = k[1] / total[1]
    return [total, probability]


def classifier(data, test):
    total = data[0]
    probability = data[1]
    n = len(test[0])

    probability2 = [log(total[0] / (total[0] + total[1])), log(total[1] / (total[0] + total[1]))]
    for i in range(n):
        for j in range(2):
            if test[i] in probability[i]:
                if probability[i][test[i]][j] != 0:
                    probability2[j] += log(probability[i][test[i]][j])
    if probability2[1] > probability2[0]:
        return 1
    else:
        return 0
