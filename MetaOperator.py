import random as rm
import numpy as np


class MetaOperator():
    def __init__(self): pass

    # do ta valed migire va crossover mizane vase adad sahih
    def singlePointIntegerCrossover(self, p_1, p_2):
        length = len(p_1)
        i1 = rm.randrange(length-1)
        y1 = self.IntegerUnique(p_1[:i1] + p_2[i1:], i1)
        y2 = self.IntegerUnique(p_2[:i1] + p_1[i1:], i1)
        return [y1, y2]

    # 2 ta valed migire va crossover mizane vase binary vagti ke bayad p ta 1 dashte bashim
    def pSinglePointBinaryCrossover(self, p_1, p_2):
        length = len(p_1)
        i1 = rm.randrange(length-1)
        y1 = self.pBinaryUnique(p_1[:i1] + p_2[i1:], p_1.count(1))
        y2 = self.pBinaryUnique(p_2[:i1] + p_1[i1:], p_1.count(1))
        return [y1, y2]

    # 2 ta valed migire va crossover mizane vase binary vagti ke mohem nist chand 1 dashte bashim
    def singlePointBinaryCrossover(self, p_1, p_2):
        length = len(p_1)
        i1 = rm.randrange(length-1)
        y1 = p_1[:i1] + p_2[i1:]
        y2 = p_2[:i1] + p_1[i1:]
        return [y1, y2]

    # 1 valed migire va mutation mizane
    def mutation(self, p):
        length = len(p)
        i1 = rm.randrange(length-1)
        i2 = rm.randrange(length-1)
        tmp = p[i1]
        p[i1] = p[i2]
        p[i2] = tmp
        return p

    # 1 list ke [0]'sh javab va [1]'sh Objective hast ro migire va bar asas objective moratab mikone
    # yani sol[i][0] va sol[i][1]
    def mySort(self, sol):
        return sorted(sol, key=lambda l: l[1])

    def IntegerUnique(self, x, ii):
        xj = x[:ii].copy()
        for t in range(ii):
            x[t] = 0
        a = np.in1d(x, xj)
        for i in np.where(a)[0].tolist():
            x[i] = 0
        x[:ii] = xj
        tmp = np.random.permutation(len(x)).tolist()
        tmp = [t + 1 for t in tmp]
        tmp = set(tmp) - set(x.copy())
        for i in np.where(a)[0].tolist():
            x[i] = tmp.pop()
        return x

    def pBinaryUnique(self, x, p):
        le = x.count(1)

        while le != p:
            if le < p:
                for i in range(p-le):
                    x[rm.choice(np.where([not x[i] for i in range(len(x))])[0].tolist())] = 1
                    le += 1
            elif le > p:
                for i in range(le-p):
                    x[rm.choice(np.where(x)[0].tolist())] = 0
                    le -= 1
        return x
