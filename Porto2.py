import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import GradientBoostingClassifier
pd.set_option('display.max_columns', 380)
pd.set_option('display.width', 140)
pd.set_option('display.max_rows', 380)

#train
trn = pd.read_csv('~/Porto/train.csv')
target1 = trn.target
trn = trn.drop(['id','target'],axis=1)
cols = list(trn.columns)
for c in cols:
    if trn.loc[trn[c]==-1.0,c].shape[0] > 0:
        trn.loc[trn[c]==-1.0,c] = trn[c].median()

trn.loc[trn['ps_car_03_cat']==-1.0,'ps_car_03_cat'] = 0

#test
test = pd.read_csv('~/Porto/test.csv')
ids = test.id
test = test.drop(['id'],axis=1)
cols1 = test.columns
for c in cols1:
    if test.loc[test[c]==-1.0,c].shape[0] > 0:
        test.loc[test[c]==-1.0,c] = test[c].median()

test.loc[test['ps_car_03_cat']==-1.0,'ps_car_03_cat'] = 0
tra = np.array(trn,order='C',copy=False)
tar = np.array(target1,order='C',copy=False)
tes = np.array(test,order='C',copy=False)

#model
model = GradientBoostingClassifier(n_estimators=150,random_state=29,verbose=2)
model = model.fit(tra,tar)
pred = model.predict_proba(tes)
pred1 = [pred[i][1] for i in range(len(pred))]
sol1 = pd.DataFrame(ids,columns=['id'])
sol1['target'] = pred1
sol1.to_csv('Porto2.csv',index=False)


