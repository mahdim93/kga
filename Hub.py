import numpy as np
import pandas as pd
import math as mh


class pHub():
    def __init__(self, dataFile):
        self.readData(dataFile)

    def readData(self, dataFile):
        f = open(dataFile)
        self.n = int(f.readline())
        self.coordination = []
        for i in range(self.n):
            self.coordination.append(list(map(float, f.readline().split())))
        self.flow = [[0]*self.n]*self.n
        for i in range(self.n):
            self.flow[i] = f.readline().split()
            self.flow[i] = list(map(float, self.flow[i]))
        self.p = int(f.readline())
        self.cc = float(f.readline())
        self.ct = float(f.readline())
        self.cd = float(f.readline())
        f.close()
        self.distance = self.calcDistance(self.coordination)
        self.fd = self.calcFlowDist(self.flow, self.distance)

    def getCoord(self):
        return self.coordination

    def getP(self):
        return self.p

    def getNodes(self):
        return self.n

    def getFlow(self):
        return self.flow

    def getDistance(self):
        return self.distance

    def Objective(self, sol):
        obj = 0
        x = self.decode(sol)
        for i in range(0, self.n):
            for k in range(0, self.n):
                if x[i] == k+1:
                    obj = obj + self.fd[i][k]
                for j in range(0, self.n):
                    for l in range(0, self.n):
                        if x[i] == k+1 and x[j] == l+1:
                            obj = obj + self.ct*self.flow[i][j]*self.distance[k][l]
        return obj

    def decode(self, sol):
        s = [[None]*self.p]*2
        x = [None]*self.n
        s[0] = np.where(sol[0])[0].tolist()
        k = 0
        for i in range(0, self.p):
            for j in range(0, self.n):
                if (s[0][i]+1) == sol[1][j]:
                    s[1][k] = j
                    k = k+1
        s = np.array(s).T.tolist()
        df = pd.DataFrame(s)
        df = df.sort_values(by=1, ascending=1)
        s = df.values.tolist()
        s = np.array(s).T.tolist()
        s[0] = [s[0][t]+1 for t in range(self.p)]
        for i in range(0, self.p):
            for j in range(0, s[1][-1]):
                if j <= s[1][i] and x[sol[1][j]-1] is None:
                    x[sol[1][j]-1] = s[0][i]
        for j in range(s[1][-1], self.n):
            x[sol[1][j]-1] = sol[1][s[1][-1]]
        for i in range(0, self.p):
            x[s[0][i]-1] = s[0][i]
        return x

    def calcDistance(self, c):
        dist = [[0]*self.n]*self.n
        for i in range(0, self.n):
            t = [0]*self.n
            for j in range(0, self.n):
                tmp = mh.pow(c[i][0]-c[j][0], 2) + mh.pow(c[i][1]-c[j][1], 2)
                t[j] = mh.sqrt(tmp)
            dist[i] = t
        return dist

    def calcFlowDist(self, f, d):
        fd = [[0]*self.n]*self.n
        for i in range(0, self.n):
            t = [0]*self.n
            for k in range(0, self.n):
                tmp = 0
                for j in range(0, self.n):
                    tmp = tmp + self.cc*f[i][j]*d[i][k] + self.cd*f[j][i]*d[i][k]
                t[k] = tmp
            fd[i] = t
        return fd

    def hubCrossover(self, p):
        dec = self.decode(p)
        f = [0]*self.n
        sn = [0]*self.n
        sh = [0]*self.n
        ll = [[t+1] for t in np.where(p[0])[0].tolist()]
        for i in p[1]:
            for j in range(self.n):
                f[i-1] += (self.flow[i-1][j]+self.flow[j][i-1])
            for k in np.where(p[0])[0].tolist():
                if (self.distance[i-1][k]*f[i-1] <= sn[i-1] or sn[i-1] == 0) and dec[i-1] != k+1:
                    sn[i-1] = k+1
                if dec[i-1] == k+1 and self.distance[i-1][k]*f[i-1] >= sh[k]:
                    sh[k] = i
        for k in range(self.p):
            for i in range(self.n):
                if dec[i] == ll[k][0] and (i+1) != ll[k][0] and sh[ll[k][0]-1] != i+1:
                    ll[k].append(i+1)
                elif dec[i] == ll[k][0] and (i+1) != ll[k][0] and sh[ll[k][0]-1] == i+1:
                    for j in range(self.p):
                        if sn[i] == np.where(p[0])[0].tolist()[j]+1:
                            ll[j].append(i+1)
        ll[0].reverse()
        for k in range(1, self.p):
            ll[k].reverse()
            ll[0].extend(ll[k])
        return [p[0], ll[0]]
