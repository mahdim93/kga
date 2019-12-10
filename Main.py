import Hub
import MetaOperator
import random as rm
import sklearn.cluster as cr
import numpy
import time
import csv
import math

def randCreatInitPop(hub):
    nodes = hub.getNodes()
    p = hub.getP()
    sol = [[0]*nodes]*2
    while sum(sol[0][:]) < p:
        sol[0][rm.randint(0, nodes-1)] = 1
    tmp = numpy.random.permutation(nodes).tolist()
    sol[1] = [t + 1 for t in tmp]
    return sol


def KmeansCreatInitPop(hub):
    coord = hub.getCoord()
    k = hub.getP()
    n = hub.getNodes()
    d = hub.getDistance()
    f = h.getFlow()
    tmp = [0]*n
    sol = []
    solc = [0]*k
    solp = [0]*k
    soln = [0]*k
    sol.append(tmp.copy())
    sol.append(tmp.copy())
    kmeans = cr.KMeans(n_clusters=k, random_state=None).fit(coord)
    labels = kmeans.labels_.tolist()
    print(labels)
    for i in range(n):
        tmp = 0
        for j in range(n):
            if labels[j] == labels[i]:
                soln[labels[i]] += 1
                for l in range(n):
                    tmp += d[i][j]*(f[j][l]+f[l][j])
        if tmp < solc[labels[i]] or solc[labels[i]] == 0:
            solc[labels[i]] = tmp
            solp[labels[i]] = i
    t = 0
    soln = [int(math.sqrt(t)) for t in soln]
    for i in range(k):
        sol[0][solp[i]] = 1
        sol[1][soln[i]+t-1] = solp[i]
        v = 1
        for j in range(n):
            if labels[j] == labels[solp[i]] and j != solp[i]:
                sol[1][soln[i]+t-v-1] = j
                v += 1
        t += soln[i]
    sol[1] = [t+1 for t in sol[1]]
    return sol


dataFile = '50.5'

h = Hub.pHub(dataFile)
mo = MetaOperator.MetaOperator()

np = 50
mi = 100
pc = 30
pm = 20

pop = []
cpop = []
cpop2 = []
mpop = []
besti = []
archive = []

start_time = time.time()

for i in range(np):
    sol = randCreatInitPop(h)
    # sol = KmeansCreatInitPop(h)
    obj = h.Objective(sol)
    pop.append([sol, obj])
pop = mo.mySort(pop[:])
init_t = ['Initial time', time.time() - start_time]
start_time = time.time()
besti.append(pop[0].copy())
for ii in range(1, mi+1):
    cpop.clear()
    mpop.clear()
    cpop2.clear()
    archive.clear()
    for i in range(int(pc)):
        x1 = pop[rm.randint(0, np-1)][0].copy()
        x2 = pop[rm.randint(0, np-1)][0].copy()
        sol = mo.singlePointIntegerCrossover(x1[1].copy(), x2[1].copy())
        obj1 = h.Objective([x1[0], sol[0]])
        obj2 = h.Objective([x2[0], sol[1]])
        cpop.append([[x1[0], sol[0]].copy(), obj1])
        cpop.append([[x2[0], sol[1]].copy(), obj2])
        sol = mo.pSinglePointBinaryCrossover(x1[0].copy(), x2[0].copy())
        obj1 = h.Objective([sol[0], x1[1]])
        obj2 = h.Objective([sol[1], x2[1]])
        cpop.append([[sol[0], x1[1]].copy(), obj1])
        cpop.append([[sol[1], x2[1]].copy(), obj1])
    for i in range(pm):
        x = pop[rm.randint(0, np-1)][0].copy()
        sol = mo.mutation(x[0].copy())
        obj = h.Objective([sol, x[1]])
        mpop.append([[sol, x[1]].copy(), obj])
        sol = mo.mutation(x[1].copy())
        obj = h.Objective([x[0], sol])
        mpop.append([[x[0], sol].copy(), obj])
#    for i in range(int(pc/2)):
#        x = pop[rm.randint(0, np-1)][0].copy()
#        sol = h.hubCrossover(x.copy())
#        obj = h.Objective(sol)
#        cpop2.append([sol.copy(), obj])
    archive.extend(pop[:])
    archive.extend(cpop[:])
    archive.extend(mpop[:])
#    archive.extend(cpop2[:])
    archive = mo.mySort(archive)
    pop.clear()
    pop = archive[:np]
    besti.append(pop[0].copy())
    print("Iteration =", ii, "Best Fitness =", besti[ii])
f = open(dataFile+"answer.csv", 'w')
iter_t = ['Iteration time', time.time() - start_time]
writer = csv.writer(f, lineterminator='\n')
writer.writerow(init_t)
writer.writerow(iter_t)
for i in range(mi+1):
    writer.writerow(besti[i])
f.close()
