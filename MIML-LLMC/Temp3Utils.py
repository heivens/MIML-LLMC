import numpy as np
from scipy.io import loadmat
import os
from tqdm import tqdm
from sklearn.metrics import euclidean_distances as eucl


def get_index(num_bags, para_k=10, seed=0):
    if seed == 0:
        temp_rand_idx = np.random.permutation(num_bags)
    else:
        r = np.random.RandomState(seed)
        temp_rand_idx = r.permutation(num_bags)

    temp_fold = int(np.ceil(num_bags / para_k))
    ret_tr_idx = {}
    ret_te_idx = {}
    for i in range(para_k):
        temp_tr_idx = temp_rand_idx[0: i * temp_fold].tolist()
        temp_tr_idx.extend(temp_rand_idx[(i + 1) * temp_fold:])
        ret_tr_idx[i] = temp_tr_idx
        ret_te_idx[i] = temp_rand_idx[i * temp_fold: (i + 1) * temp_fold].tolist()
    return ret_tr_idx, ret_te_idx


class MyData:
    def __init__(self, path, negative_label=0, measure='ave'):
        """获得padding后的包集合, 包标签集合, 所有原始包的距离矩阵"""
        self.ori_bags, self.labels, self.num_bags, self.num_ins = self.__load_bags_labels(path, negative_label=negative_label)
        self.dataset_name = path.split('/')[-1].split('.')[0]
        self.ins_len = self.ori_bags[0].shape[1]
        self.bag_size = []
        self.padded_bags = self.__pad_bags()
        self.dis_matrix = self.__get_matrix(measure)

    def __get_matrix(self, measure):
        dis_path = '../Distance/' + self.dataset_name + '_' + str(measure) + '.npy'
        if os.path.exists(dis_path):
            # print('Load Pre-computed Distance Matrix')
            dis_matrix = np.load(dis_path)
            return dis_matrix
        else:
            dis_matrix = np.zeros((self.num_bags, self.num_bags))
            if measure == 'ave':
                for i in tqdm(range(self.num_bags), desc='Computing Distance Matrix of ' + measure):
                    for j in range(i, self.num_bags):
                        dis = self.__ave_hausdorff(i, j)
                        dis_matrix[i, j] = dis
                        dis_matrix[j, i] = dis
                np.save(dis_path, dis_matrix)
                return dis_matrix
            elif measure == 'max':
                for i in tqdm(range(self.num_bags), desc='Computing Distance Matrix of ' + measure):
                    for j in range(i, self.num_bags):
                        dis = self.__max_hausdorff(i, j)
                        dis_matrix[i, j] = dis
                        dis_matrix[j, i] = dis
                np.save(dis_path, dis_matrix)
                return dis_matrix
            elif measure == 'min':
                for i in tqdm(range(self.num_bags), desc='Computing Distance Matrix of ' + measure):
                    for j in range(i, self.num_bags):
                        dis = self.__min_hausdorff(i, j)
                        dis_matrix[i, j] = dis
                        dis_matrix[j, i] = dis
                np.save(dis_path, dis_matrix)
                return dis_matrix

    def __pad_bags(self):
        for bag in self.ori_bags:
            self.bag_size.append(len(bag))
        max_bag_size = np.max(self.bag_size)
        padded_bags = []
        for i, bag in enumerate(self.ori_bags):
            pad_size = max_bag_size - self.bag_size[i]
            if pad_size:
                pad = np.zeros((pad_size, self.ins_len))
                padded_bag = np.vstack((bag, pad))
                padded_bags.append(padded_bag)
            else:
                padded_bags.append(bag)
        padded_bags = np.array(padded_bags)
        return padded_bags

    def __load_bags_labels(self, path, negative_label):
        data = loadmat(path)
        # print(data['bags'])
        num_bags = len(data['bags'])
        bags = []
        num_ins = 0
        for i in range(num_bags):
            bags.append(data['bags'][i][0])  # .tolist())
            num_ins += len(data['bags'][i][0])
        # bags = np.array(bags, dtype=object)
        labels = np.array(data['labels'])
        labels = np.where(labels > 0, labels, negative_label)
        return bags, labels, num_bags, num_ins

    def __ave_hausdorff(self, i, j):
        if i == j:
            return 0
        temp_1 = np.sum(np.min(eucl(self.ori_bags[i], self.ori_bags[j]), axis=1))
        temp_2 = np.sum(np.min(eucl(self.ori_bags[j], self.ori_bags[i]), axis=1))
        result = (temp_1 + temp_2) / (self.ori_bags[i].shape[0] + self.ori_bags[j].shape[0])
        return result

    def __max_hausdorff(self, i, j):
        if i == j:
            return 0
        temp_1 = np.max(np.min(eucl(self.ori_bags[i], self.ori_bags[j]), axis=1))
        temp_2 = np.max(np.min(eucl(self.ori_bags[j], self.ori_bags[i]), axis=1))
        result = np.max((temp_1, temp_2))
        return result

    def __min_hausdorff(self, i, j):
        if i == j:
            return 0
        result = np.min(eucl(self.ori_bags[i], self.ori_bags[j]))
        return result


if __name__ == '__main__':
    path = '../DataSets/Unnormalized/reuters_MIML.mat'
    mydata = MyData(path, measure='max')