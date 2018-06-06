import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import average_precision_score
from load_data import DataLoader


class TableNode(object):
    def __init__(self, index):
        self.val = index
        self.buckets = {}


class LSH:
    def __init__(self):
        self.num = 0

    def dist(self, p1, p2):
        """get euclidean distance of 2 vectors"""

        p1, p2 = np.array(p1), np.array(p2)
        return np.sqrt(np.sum(np.square(p1 - p2)))

    def genPara(self, d, r):
        """
        :param d: length of data vector
        :return: a
        """

        a = []
        for i in range(d):
            a.append(random.gauss(0, 1))
        b = random.uniform(0,r)

        return a, b

    def gen_e2LSH_family(self, d, k, r):
        """
        :param d: length of data vector
        :param k:
        :param r:
        :return: a list of parameters a
        """
        result = []
        for i in range(k):
            result.append(self.genPara(d, r))

        return result

    def gen_HashVals(self, e2LSH_family, p, r):
        """
        :param e2LSH_family: include k hash funcs(parameters)
        :param p: data vector
        :param r:
        :return hash values: a list
        """

        # hashVals include k values
        hashVals = []

        for hab in e2LSH_family:
            hashVal = (np.inner(hab[0], p) + hab[1]) // r
            hashVals.append(hashVal)

        return hashVals

    def H2(self, hashVals, fpRand, k, C):
        """
        :param hashVals: k hash vals
        :param fpRand: ri', the random vals that used to generate fingerprint
        :param k, C: parameter
        :return: the fingerprint of (x1, x2, ..., xk), a int value
        """
        return int(sum([(hashVals[i] * fpRand[i]) for i in range(k)]) % C)

    def e2LSH(self, dataSet, k, L, r, tableSize):
        """
        generate hash table
        * hash table: a list, [node1, node2, ... node_{tableSize - 1}]
        ** node: node.val = index; node.buckets = {}
        *** node.buckets: a dictionary, {fp:[p1, ..], ...}
        :param dataSet: a set of vector(list)
        :param k:
        :param L:
        :param r:
        :param tableSize:
        :return: 3 elements, hash table, hash functions, fpRand
        """

        hashTable = [TableNode(i) for i in range(tableSize)]

        d = len(dataSet[0])
        n = len(dataSet)

        C = pow(2, 32) - 5
        hashFuncs = []
        fpRand = [random.randint(-10, 10) for i in range(k)]

        for times in range(L):

            e2LSH_family = self.gen_e2LSH_family(d, k, r)

            # hashFuncs: [[h1, ...hk], [h1, ..hk], ..., [h1, ...hk]]
            # hashFuncs include L hash functions group, and each group contain k hash functions
            hashFuncs.append(e2LSH_family)

            for dataIndex in range(n):

                # generate k hash values
                hashVals = self.gen_HashVals(e2LSH_family, dataSet[dataIndex], r)

                # generate fingerprint
                fp = self.H2(hashVals, fpRand, k, C)

                # generate index
                index = fp % tableSize

                # find the node of hash table
                node = hashTable[index]

                # node.buckets is a dictionary: {fp: vector_list}
                if fp in node.buckets:

                    # bucket is vector list
                    bucket = node.buckets[fp]

                    # add the data index into bucket
                    bucket.append(dataIndex)

                else:
                    node.buckets[fp] = [dataIndex]

        return hashTable, hashFuncs, fpRand

    def score(self, result, dataLabel, K):
        sum = np.zeros(19)
        for index in result:
            sum += dataLabel[index]
        return sum / K


    def nn_search(self, dataSet, dataLabel, imgId, querySet, queryLabel, k, L, r, tableSize, K, c):
        """
        :param dataSet:
        :param querySet:
        :param k:
        :param L:
        :param r:
        :param tableSize:
        :return: the data index that similar with query
        """

        tmp = self.e2LSH(dataSet, k, L, r, tableSize)
        C = pow(2, 32) - 5

        hashTable = tmp[0]
        hashFuncGroups = tmp[1]
        fpRand = tmp[2]
        true = []
        scores = []
        avg_dist = 0
        search_time = 0
        for i in range(querySet.shape[0]):
            t1 = time.time()
            query = querySet[i]
            result = set()
            img_ids = set()
            foundK = False
            count = 0
            total_dist = 0
            for hashFuncGroup in hashFuncGroups:

                # get the fingerprint of query
                queryFp = self.H2(self.gen_HashVals(hashFuncGroup, query, r), fpRand, k, C)

                # get the index of query in hash table
                queryIndex = queryFp % tableSize

                # get the bucket in the dictionary
                if queryFp in hashTable[queryIndex].buckets:
                    # count += len(hashTable[queryIndex].buckets[queryFp])
                    for p in hashTable[queryIndex].buckets[queryFp]:
                        if img_ids.__contains__(imgId[p]):
                            continue
                        img_ids.add(imgId[p])
                        distance = self.dist(dataSet[p], query)
                        count += 1
                        if distance <= c * r:
                            result.add(p)
                            total_dist += distance
                            if len(result) == K:
                                foundK = True
                                break
                    # result.update(hashTable[queryIndex].buckets[queryFp])

                if foundK:
                    result = list(result)[:K]
                    break

                if count >= 3 * L and len(result) == 0:
                    break

            if foundK:
                avg_dist += total_dist / K
                search_time += count
                true.append(queryLabel[i])
                scores.append(self.score(result, dataLabel, K))
                t2 = time.time()
                print("%d: %d seconds, one patch done" % (i, t2 - t1))
            else:
                t2 = time.time()
                print("%d: %d seconds, no neighbor is found" % (i, t2 - t1))

        mAP = average_precision_score(true, scores, average="micro")

        return search_time, avg_dist, mAP
