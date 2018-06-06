from load_data import DataLoader
from model import LSH
import numpy as np


def dist(v1, v2):
    """get euclidean distance of 2 vectors"""

    v1, v2 = np.array(v1), np.array(v2)
    return np.sqrt(np.sum(np.square(v1 - v2)))


def main():
    dataDir = '/Users/Blair/PycharmProjects/CSE547/data'
    dataTypes = ["train2014", "val2014", "test2014"]
    for dataType in dataTypes[:2]:
        loader = DataLoader(dataDir, dataType)
        loader.load()
        if dataType == dataTypes[0]:
            train_dt = loader.feats
            train_lab = loader.label
            train_id = loader.id
        elif dataType == dataTypes[1]:
            val_dt = loader.feats
            val_lab = loader.label
        else:
            test_dt = loader.feats
            test_lab = loader.label

    model = LSH()
    search_time, avg_dist, mAP = model.nn_search(train_dt, train_lab, train_id, val_dt, val_lab, 20, 5, 100, 20, 10, 1.5)
    print(search_time)
    print(avg_dist)
    print(mAP)


if __name__ == '__main__':
    main()