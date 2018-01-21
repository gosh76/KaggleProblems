#To write ML program in python for AllState problem.
import numpy as np
import pandas as pd
import csv
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
listd = []
def trg(n):
    trg1 = pd.read_csv('train.csv')
    target = list(trg1.loss)
    trg1 = trg1.drop(['id','loss'],axis=n)
    list1 = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    dict0 = dict(zip((list1),(x + 1 for x in range(0,len(list1)))))
    dict1 = dict(zip((list(trg1.cat109.unique())),(x + 1 for x in range(0,len(list(trg1.cat109.unique()))))))
    dict2 = dict(zip((list(trg1.cat110.unique())),(x + 1 for x in range(0,len(list(trg1.cat110.unique()))))))
    dict3 = dict(zip((list(trg1.cat112.unique())),(x + 1 for x in range(0,len(list(trg1.cat112.unique()))))))
    dict4 = dict(zip((list(trg1.cat113.unique())),(x + 1 for x in range(0,len(list(trg1.cat113.unique()))))))
    dict5 = dict(zip((list(trg1.cat116.unique())),(x + 1 for x in range(0,len(list(trg1.cat116.unique()))))))
    cols = list(trg1.columns)
    trg1[cols[0:108]] = trg1[cols[0:108]].applymap(lambda z:dict0[z])
    trg1[cols[108]] = trg1[cols[108]].map(dict1)
    trg1[cols[109]] = trg1[cols[109]].map(dict2)
    trg1[cols[110]] = trg1[cols[110]].map(dict0)
    trg1[cols[111]] = trg1[cols[111]].map(dict3)
    trg1[cols[112]] = trg1[cols[112]].map(dict4)
    trg1[cols[113:115]] = trg1[cols[113:115]].applymap(lambda z:dict0[z])
    trg1[cols[115]] = trg1[cols[115]].map(dict5)
    trg2 = np.array(trg1)
    target1 = np.array(target)
    listd.append(trg2)
    listd.append(target1)

trg(1)
def modelling(n):
    model = GradientBoostingRegressor(loss='lad',n_estimators=300,learning_rate=0.1,criterion='friedman_mse',min_samples_split=850,min_samples_leaf=200,max_depth=20,\
    subsample=0.95,verbose=2,random_state=n)
    model = model.fit(listd[0],listd[1])
    f = open('f1.p','w')
    pickle.dump(model,f)
    f.close()

modelling(59)

#TESTER

import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import GradientBoostingRegressor
import pickle
def testing(n):
    f = open('f1.p','r')
    model = pickle.load(f)
    f.close()
    test1 = pd.read_csv('test.csv')
    ids = list(test1.id)
    test1 = test1.drop(['id'],axis=n)
    list1 = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    dict0 = dict(zip((list1),(x + 1 for x in range(0,len(list1)))))
    dict1 = dict(zip((list(test1.cat109.unique())),(x + 1 for x in range(0,len(list(test1.cat109.unique()))))))
    dict2 = dict(zip((list(test1.cat110.unique())),(x + 1 for x in range(0,len(list(test1.cat110.unique()))))))
    dict3 = dict(zip((list(test1.cat112.unique())),(x + 1 for x in range(0,len(list(test1.cat112.unique()))))))
    dict4 = dict(zip((list(test1.cat113.unique())),(x + 1 for x in range(0,len(list(test1.cat113.unique()))))))
    dict5 = dict(zip((list(test1.cat116.unique())),(x + 1 for x in range(0,len(list(test1.cat116.unique()))))))
    cols = list(test1.columns)
    test1[cols[0:108]] = test1[cols[0:108]].applymap(lambda z:dict0[z])
    test1[cols[108]] = test1[cols[108]].map(dict1)
    test1[cols[109]] = test1[cols[109]].map(dict2)
    test1[cols[110]] = test1[cols[110]].map(dict0)
    test1[cols[111]] = test1[cols[111]].map(dict3)
    test1[cols[112]] = test1[cols[112]].map(dict4)
    test1[cols[113:115]] = test1[cols[113:115]].applymap(lambda z:dict0[z])
    test1[cols[115]] = test1[cols[115]].map(dict5)
    test2 = np.array(test1)
    ids1 = np.array(ids)
    pred = model.predict(test2)
    out1 = pd.DataFrame(ids1,columns=['id'])
    out1['loss'] = pred
    out1.to_csv('AllstateF26.csv',index=False)

testing(1)

