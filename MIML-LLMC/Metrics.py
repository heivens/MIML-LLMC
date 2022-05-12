from sklearn.metrics import label_ranking_loss
import numpy as np


# 需要传入负类的标签(0或-1)
def HammingLoss(y_true, output, negative_po=0):
    if negative_po == 0:
        y_pred = np.where(output > 0.5, 1., 0.)
    else:
        y_pred = np.where(output > 0., 1., -1.)
    num_ins, num_class = y_pred.shape
    miss_pairs = np.sum(y_pred != y_true)
    return miss_pairs / (num_ins * num_class)


# 负类标签可为0或-1, 结果都跟matlab源码无异
def OneErrorLoss(y_true, output):
    loss = 0
    for i in range(len(y_true)):
        if y_true[i][np.argmax(output[i])] != 1:
            loss += 1
    return loss / len(y_true)


# 负类标签可为0或-1, 结果都跟matlab源码无异
def CoverageError(y_true, output):
    num_ins, num_class = output.shape
    # print(num_ins)
    label = []
    label_size = []
    for i in range(num_ins):
        # print(i)
        temp = y_true[i]
        label_size.append(int(np.sum(temp == np.ones(num_class))))
        idx = []
        for j in range(num_class):
            if temp[j] == 1:
                idx.append(j)
        label.append(idx)
    cover = 0
    for i in range(num_ins):
        temp = output[i]
        index = np.argsort(temp)
        temp_min = num_class + 1
        for m in range(label_size[i]):
            loc = np.argwhere(index == label[i][m])[0, 0]
            if loc < temp_min:
                temp_min = loc + 1
        cover += num_class - temp_min + 1
    return ((cover / num_ins) - 1) / y_true.shape[1]


# 负类标签可为0或-1, 结果都跟matlab源码无异
def RankingLoss(y_true, output):
    # y_true为(0, 1)和(-1, 1)时得到的结果一样
    y_true = np.where(y_true > 0., 1., 0.)
    return label_ranking_loss(y_true, output)  # 此函数只支持负类为0的情况,所以需要先转换


# 负类标签可为0或-1, 结果都跟matlab源码无异
def AveragePrecision(y_true, output):
    num_ins, num_class = output.shape
    label = []
    label_size = []
    for i in range(num_ins):
        temp = y_true[i]
        label_size.append(int(np.sum(temp == np.ones(num_class))))
        idx = []
        for j in range(num_class):
            if temp[j] == 1:
                idx.append(j)
        label.append(idx)
    aveprec = 0
    for i in range(num_ins):
        temp = output[i]
        index = np.argsort(temp)
        indicator = np.zeros(num_class)
        for m in range(label_size[i]):
            loc = np.argwhere(index == label[i][m])[0][0]
            indicator[loc] = 1
        summary = 0
        for m in range(label_size[i]):
            loc = np.argwhere(index == label[i][m])[0, 0]
            summary += np.sum(indicator[loc: num_class]) / (num_class - loc)
        aveprec += summary / label_size[i]
    return aveprec / num_ins


def AllFive(y_true, output):
    hamming = HammingLoss(y_true, output)
    one_error = OneErrorLoss(y_true, output)
    coverage = CoverageError(y_true, output)
    ranking = RankingLoss(y_true, output)
    aver_per = AveragePrecision(y_true, output)
    return hamming, one_error, coverage, ranking, aver_per
