import numpy as np
import pandas as pd
import csv
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
pd.set_option('display.max_columns', 380)
pd.set_option('display.width', 140)
pd.set_option('display.max_rows', 380)

#train
train = pd.read_csv('~/Desktop/sfolder/benztrain.csv')
target = train.y
train = train.drop(['y','ID'],axis=1)
listd = list(train.dtypes[train.dtypes==object].index)
listg = []
for x in range(len(listd)):
    listg.append(dict(zip(list(train[listd[x]].unique()),range(len(train[listd[x]].unique())))))

for a,b in zip(listd,range(len(listg))):
    train.loc[:,a] = train.loc[:,a].map(lambda x:listg[b][x])

tra = np.array(train)
tar = np.array(target)
model = XGBRegressor(max_depth=3,learning_rate=0.1,n_estimators=100,silent=False,objective='reg:linear',nthread=16,gamma=0,min_child_weight=1,max_delta_step=\
0,subsample=1,colsample_bytree=0.8,colsample_bylevel=0.8,reg_alpha=0.3,reg_lambda=1,scale_pos_weight=1,base_score=0.3,seed=67,missing=None)
model = model.fit(tra[0:3800],tar[0:3800])
print(r2_score(tar[3800:],model.predict(tra[3800:])))

#test
test = pd.read_csv('~/Desktop/sfolder/benztest.csv')
ids = test.ID
test = test.drop(['ID'],axis=1)
listd1 = list(test.dtypes[test.dtypes==object].index)
listg1 = []
for x in range(len(listd1)):
    listg1.append(dict(zip(list(test[listd1[x]].unique()),range(len(test[listd1[x]].unique())))))

for a,b in zip(listd1,range(len(listg1))):
    test.loc[:,a] = test.loc[:,a].map(lambda x:listg1[b][x])

tes = np.array(test)
pred = model.predict(tes)
submission = pd.DataFrame(ids,columns=['ID'])
submission['y'] = pred
submission.to_csv('benz4.csv',index=False)

