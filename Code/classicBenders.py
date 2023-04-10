'''
Problem:
MINIMIZE      \sum_{i \in C} \sum_{j \in F} c_{ij} x_{ij} + \sum_{j \in F} f_j y_j
s.t.          \sum_{j \in F} x_{ij} = 1, \forall i \in C
              0 <= x_{ij} <= y_j, \forall i \in C, \forall j in F
              x binary, y binary
'''
import numpy as np
from gurobipy import *
import time

def subProblem(y):
    #y is a fixed element
    submodel = Model("subproblem")
    x = {(i,j):submodel.addVar(lb=0,vtype=GRB.CONTINUOUS) for i in range(C) for j in range(F)}
    constrAlpha = {}
    constrBeta = {}
    for i in range(C):
        constrAlpha[i] = submodel.addConstr(sum(x[i,j] for j in range(F))==1)
    for i in range(C):
        for j in range(F):
            constrBeta[i,j] = submodel.addConstr(x[i,j] <= y[j])
    obj = sum([cost[i,j]*x[i,j] for j in range(F) for i in range(C)]) + sum([f[j]*y[j] for j in range(F)])
    submodel.setObjective(obj,sense=GRB.MINIMIZE)
    submodel.update()
    submodel.Params.OutputFlag = 0
    submodel.Params.InfUnbdInfo = 1
    submodel.Params.DualReductions = 0
    submodel.optimize()
    #submodel.write("sub.lp")

    #get dual value
    alpha = {}
    beta = {}
    if submodel.status == GRB.OPTIMAL:
        for i in range(C):
            alpha[i] = constrAlpha[i].pi
            for j in range(F):
                beta[i,j] = constrBeta[i,j].pi
        return submodel.objVal,alpha,beta,[x[i,j].x for i in range(C) for j in range(F)],submodel.status
    else:
        for i in range(C):
            alpha[i] = constrAlpha[i].FarkasDual
            for j in range(F):
                beta[i,j] = constrBeta[i,j].FarkasDual

        return -float("inf"),alpha,beta,[],submodel.status

def setupMasterproblem():
    model = Model("Masterproblem")
    z = model.addVar(vtype=GRB.CONTINUOUS,name="z")
    y = {j:model.addVar(vtype=GRB.BINARY,name = str(j)) for j in range(F)}

    model.setObjective(z+sum([f[j]*y[j] for j in range(F)]),sense=GRB.MINIMIZE)
    model.addConstr(quicksum(y[j] for j in range(F)) >= 1)
    model.update()
    model.Params.OutputFlag = 0
    model.Params.LazyConstraints = 1
    model.optimize()

    return model,[y[j].x for j in range(F)]

def updateMaster(model,optCuts_alpha,optCuts_beta,fesCuts_alpha,fesCuts_beta):
    dual = 0
    #add optimal cut
    if optCuts_alpha:
        dual = sum(optCuts_alpha.values())
        for j in range(F):
            for i in range(C):
                dual += optCuts_beta[i,j]*model.getVarByName(str(j))
        model.addConstr(dual <= model.getVarByName("z"))
        #model.write("master.lp")
    #add feasible cut
    else:
        dual = sum(fesCuts_alpha.values())
        for j in range(F):
            for i in range(C):
                dual += fesCuts_beta[i, j] * model.getVarByName(str(j))
        model.addConstr(dual >= 0)
    #reoptimize
    model.optimize()
    if model.status == GRB.OPTIMAL:
        return model.objVal,[model.getVarByName(str(j)).x for j in range(F)],model.getVarByName("z"),model
    else:
        print("something went wrong in ths master problem and its status is",model.status)

def Benders(eps = 0,maxiter = 5):
    UB,LB = float("inf"),-float("inf")
    iters = 0
    tol = float("inf")
    model,y = setupMasterproblem()

    while iters < maxiter and eps < tol:
        subobj,alpha,beta,x,status = subProblem(y)
        UB = min(UB,subobj)
        if status == 2:
            masterobj,y,z,model = updateMaster(model,alpha,beta,[],[])
        else:
            masterobj,y,z,model = updateMaster(model,[],[],alpha,beta)
        LB = masterobj
        tol = UB-LB
        if iters%20 == 0:
            print("ub",UB,"LB",LB,"Gap",(UB-LB)/UB)
        iters += 1
    return y,x,UB

def generateFacilityLocationData(C, F):
    #seed 42
    np.random.seed(42)
    cost =  np.random.randint(1000, size=(C, F))
    f = np.random.randint(1000, size=(F))
    for j in range(F):
        for i in range(C):
            f[j] += round(0.05*cost[i,j])
    return C, F, cost, f

def solveModelGurobi(C,F):
    m2 = Model()
    y = {j: m2.addVar(lb=0, vtype=GRB.BINARY) for j in range(F)}
    x = {(i, j): m2.addVar(lb=0, vtype=GRB.BINARY) for i in range(C) for j in range(F)}
    for i in range(C):
        m2.addConstr(sum([x[i, j] for j in range(F)]) == 1)
    for j in range(F):
        for i in range(C):
            m2.addConstr(x[i, j] <= y[j])
    obj = 0
    for j in range(F):
        obj = obj + f[j] * y[j]
        for i in range(C):
            obj += cost[i, j] * x[i, j]
    m2.setObjective(obj, sense=GRB.MINIMIZE)
    m2.update()
    m2.Params.OutputFlag = 0
    m2.optimize()
    yVal= [y[j].x for j in range(F)]
    xVal =[x[i, j].x for i in range(C) for j in range(F)]
    print(m2.objVal, yVal)
    return m2.objVal, yVal, xVal

C,F = 100,10
C, F, cost, f = generateFacilityLocationData(C, F)
start_time = time.time()
y,x,obj = Benders(maxiter=1000)
print(obj)
print("benders time:",(time.time()-start_time),"seconds")
start_time = time.time()
objgurobi,x,y = solveModelGurobi(C,F)
print(objgurobi)
print("benders time:",(time.time()-start_time),"seconds")
