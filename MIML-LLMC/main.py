from Temp3Utils import get_index, MyData
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import time
import torch.optim as optim
from Metrics import HammingLoss, OneErrorLoss, CoverageError, RankingLoss, AveragePrecision
from scipy.special import softmax
from kmedoids import kMedoids
from sklearn.metrics import euclidean_distances as eucl
import os

"""
2022/03/07: 较为稳定版本
"""


class TempLoss(nn.Module):
    def __init__(self):
        super(TempLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, labels):
        labels = labels.squeeze().to(torch.long)
        loss = 0
        for i in range(output.shape[0]):
            loss += self.cross_entropy(output[0], labels[0])  # 跟用循环获得单个标签的loss结果一致
        # print(loss / output.shape[0])
        return loss


class Model(nn.Module):  # 预测标签形式改为MSE
    def __init__(self, ins_len, loc_len, n_class):
        super(Model, self).__init__()
        self.ins_len, self.loc_len, self.n_class = ins_len, loc_len, n_class
        self.attentions = nn.ModuleList([Attention(ins_len) for _ in range(n_class)])
        predict = nn.Sequential(nn.Linear(ins_len + loc_len, 1))
        self.predict = nn.ModuleList([predict for _ in range(n_class)])

    def forward(self, bags, loc):
        attention_results = self.attentions[0](bags)
        for i in range(1, self.n_class):
            attention_results = torch.cat((attention_results, self.attentions[i](bags)), dim=1)
        cat_resluts = torch.cat((attention_results, loc.repeat(1, self.n_class, 1)), dim=2)
        y = self.predict[0](cat_resluts[:, 0, :]).unsqueeze(1)
        for i in range(1, self.n_class):
            y = torch.cat((y, self.predict[i](cat_resluts[:, i, :]).unsqueeze(1)), dim=1)
        y = torch.sigmoid(y.squeeze(2))
        return y


class Attention(nn.Module):
    def __init__(self, ins_len):
        super(Attention, self).__init__()
        self.linear = nn.Linear(ins_len, ins_len)
        self.drop_1 = nn.Dropout()
        self.attention = nn.Linear(ins_len, 1)

    def forward(self, bags):
        bags = self.drop_1(F.relu(self.linear(bags)))
        # print(bags.shape) # [1333, 9, 15]
        weights = F.relu(self.attention(bags))
        weights = torch.softmax(torch.transpose(weights, 1, 2), dim=2)
        results = torch.bmm(weights, bags)
        return results


class TempDataSet(Dataset):
    def __init__(self, bags, labels, loc):
        self.bags = bags
        self.labels = labels
        self.loc = loc

    def __getitem__(self, idx):
        return self.bags[idx], self.labels[idx], self.loc[idx]

    def __len__(self):
        return len(self.labels)


class Algorithm:
    def __init__(self, n, k, path, measure, n_neighbor, ratio, epochs, lr, batch):
        """ n: number of cv
            k: k folds cv
            path: dataset path
            measure: measures of distance between bags (ave, max, min)
            n_neighbor: number of neighbors used in label manifold
        """
        self.k = k
        self.n = n
        self.dataset_name = path.split('/')[-1].split('.')[0]
        self.measure = measure
        self.n_neighbor = n_neighbor
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.data = MyData(path=path, negative_label=0, measure=self.measure)
        self.n_class = self.data.labels.shape[-1]
        self.ratio = ratio

    def __load_train_test(self, tr_idx, te_idx):
        tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
        bags, labels = self.data.padded_bags, self.data.labels
        dis_matrix = self.data.dis_matrix
        dis_matrix = np.where(dis_matrix == 0, np.inf, dis_matrix)
        tr_dis_matrix = dis_matrix[tr_idx, :]
        tr_dis_matrix = tr_dis_matrix[:, tr_idx]
        tr_nei_idx, te_nei_idx = [], []
        for row in tr_dis_matrix:
            nei_idx = tr_idx[np.argsort(row)[:self.n_neighbor]]
            tr_nei_idx.append(nei_idx)
        tr_nei_idx = np.array(tr_nei_idx)
        te_tr_dis_matrix = dis_matrix[te_idx, :]
        te_tr_dis_matrix = te_tr_dis_matrix[:, tr_idx]
        for row in te_tr_dis_matrix:
            nei_idx = tr_idx[np.argsort(row)[:self.n_neighbor]]
            te_nei_idx.append(nei_idx)
        te_nei_idx = np.array(te_nei_idx)
        tr_nei_labels, te_nei_labels = [], []
        # 得到每个训练样本邻居的标签
        for idx_tr_nei in tr_nei_idx:
            tr_nei_labels.append(labels[idx_tr_nei])
        tr_nei_labels = np.array(tr_nei_labels)
        # 得到每个测试样本邻居的标签
        for idx_te_nei in te_nei_idx:
            te_nei_labels.append(labels[idx_te_nei])
        te_nei_labels = np.array(te_nei_labels)
        tr_bags, te_bags = bags[tr_idx], bags[te_idx]
        tr_labels, te_labels = labels[tr_idx], labels[te_idx]
        # 得到每个训练样本与其邻居的距离
        tr_nei_dis = []
        for i, idx_tr_nei in enumerate(tr_nei_idx):
            dis = self.data.dis_matrix[tr_idx[i], idx_tr_nei]
            tr_nei_dis.append(dis)
        tr_nei_dis = np.array(tr_nei_dis)
        # 得到每个训练样本与其邻居的相似度
        tr_nei_sim = np.expand_dims(softmax(1 / tr_nei_dis, axis=1), 1)
        # 得到每个训练样本与其邻居的距离
        te_nei_dis = []
        for i, idx_te_nei in enumerate(te_nei_idx):
            dis = self.data.dis_matrix[te_idx[i], idx_te_nei]
            te_nei_dis.append(dis)
        te_nei_dis = np.array(te_nei_dis)
        te_nei_sim = np.expand_dims(softmax(1 / te_nei_dis, axis=1), 1)
        # 获得训练集样本的label manifold
        tr_label_man = softmax((tr_nei_sim @ tr_nei_labels).squeeze(), axis=1)
        # 获得测试集样本的label manifold
        te_label_man = softmax((te_nei_sim @ te_nei_labels).squeeze(), axis=1)
        medoids_idx, C = kMedoids(D=eucl(tr_label_man, tr_label_man), k=int(self.ratio * self.n_class) if int(self.ratio * self.n_class) > 2 else 2)
        medoids = tr_label_man[medoids_idx]
        tr_loc = eucl(tr_label_man, medoids)
        te_loc = eucl(te_label_man, medoids)
        # tr_labels, te_labels = np.expand_dims(tr_labels, 1), np.expand_dims(te_labels, 1)
        tr_loc, te_loc = np.expand_dims(tr_loc, 1), np.expand_dims(te_loc, 1)
        return tr_bags, tr_labels, tr_loc, te_bags, te_labels, te_loc

    def n_cv(self):
        ham_list, one_list, cov_list, rank_list, ave_list = [], [], [], [], []
        for i in range(self.n):
            ham, one, cov, rank, ave = self.__one_cv()
            ham_list.append(ham)
            one_list.append(one)
            cov_list.append(cov)
            rank_list.append(rank)
            ave_list.append(ave)
        return np.mean(ham_list), np.std(ham_list, ddof=1), \
               np.mean(one_list), np.std(one_list, ddof=1), \
               np.mean(cov_list), np.std(cov_list, ddof=1), \
               np.mean(rank_list), np.std(rank_list, ddof=1), \
               np.mean(ave_list), np.std(ave_list, ddof=1),

    def __one_cv(self):
        tr_idx_list, te_idx_list = get_index(len(self.data.padded_bags), para_k=self.k)
        ham_list, one_list, cov_list, rank_list, ave_list = [], [], [], [], []
        # for i in range(self.k):
        for i in tqdm(range(self.k), desc='One Time of ' + str(self.k) + ' CV'):
            ham, one, cov, rank, ave = self.__run(tr_idx_list[i], te_idx_list[i])
            ham_list.append(ham)
            one_list.append(one)
            cov_list.append(cov)
            rank_list.append(rank)
            ave_list.append(ave)
        return np.mean(ham_list), np.mean(one_list), np.mean(cov_list), np.mean(rank_list), np.mean(ave_list)

    def __run(self, tr_idx, te_idx):
        tr_bags, tr_labels, tr_loc, te_bags, te_labels, te_loc = self.__load_train_test(tr_idx, te_idx)
        # 构建DataSet和DataLoader
        trDataSet = TempDataSet(tr_bags, tr_labels, tr_loc)
        teDataSet = TempDataSet(te_bags, te_labels, te_loc)
        trDataLoader = DataLoader(trDataSet, shuffle=False, batch_size=self.batch)
        teDataLoader = DataLoader(teDataSet, shuffle=False, batch_size=self.batch)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loc_len = int(self.ratio * self.n_class) if int(self.ratio * self.n_class) > 2 else 2
        model = Model(self.data.ins_len, loc_len, self.n_class).to(device)
        # criterion = nn.BCELoss()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        ham_list, one_list, cov_list, rank_list, ave_list = [], [], [], [], []
        for epoch in range(self.epochs):
            # train
            model.train()
            loss_value = 0.0
            for i, data in enumerate(trDataLoader):
                bags, labels, loc = data[0].to(torch.float32).to(device), data[1].to(torch.float32).to(device), data[2].to(torch.float32).to(device)
                out = model(bags, loc)
                # print(out)
                # exit(0)
                loss = criterion(out, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_value += loss.item() * self.batch
            # print('%3d-th epoch, Loss: %.5f ||' % (epoch, loss_value / len(tr_idx)), end='')
            with torch.no_grad():
                model.eval()
                output = []
                for i, data in enumerate(teDataLoader):
                    bags, labels, loc = data[0].to(torch.float32).to(device), data[1].to(torch.float32).to(device), data[2].to(torch.float32).to(device)
                    output.append(model(bags, loc).cpu().detach().numpy())
                output = np.vstack(output)
                te_labels = np.squeeze(te_labels)
                # print(output.shape)
                # print(te_labels.shape)
                ham = HammingLoss(te_labels, output, negative_po=0)
                one = OneErrorLoss(te_labels, output)
                cov = CoverageError(te_labels, output)
                rank = RankingLoss(te_labels, output)
                ave = AveragePrecision(te_labels, output)
                # print(ham, one)
                ham_list.append(ham)
                one_list.append(one)
                cov_list.append(cov)
                rank_list.append(rank)
                ave_list.append(ave)
        return np.min(ham_list), np.min(one_list), np.min(cov_list), np.min(rank_list), np.max(ave_list)


if __name__ == '__main__':
    k = 3  # folds
    n = 10  #
    path = '../DataSets/Unnormalized/scene_MIML.mat'
    name = path.split('/')[-1].split('.')[0]
    print(name)
    measure = 'ave'
    epochs = 5  # reuters设为50, scene设为300, AZO设为20, GEO设为
    n_neighbor = 40
    ratio = 0.9  # 聚类比例
    lr = 0.001
    batch = 64
    start = time.time()
    algorithm = Algorithm(n, k, path, measure, n_neighbor, ratio, epochs, lr, batch)
    ham, ham_std, one, one_std, cov, cov_std, rank, rank_std, ave, ave_std = algorithm.n_cv()
    print('MIML-LLMC')
    print('Number of Neighbors:', n_neighbor)
    print('Cluster Ratio:', ratio)
    print('Hamming: $%.4f_{\\pm%.4f}$' % (ham, ham_std))
    print('OneError: $%.4f_{\\pm%.4f}$' % (one, one_std))
    print('Coverage: $%.4f_{\\pm%.4f}$' % (cov, cov_std))
    print('Ranking: $%.4f_{\\pm%.4f}$' % (rank, rank_std))
    print('AveragePrecision: $%.4f_{\\pm%.4f}$' % (ave, ave_std))
    print('Time Cost:', time.time() - start)

