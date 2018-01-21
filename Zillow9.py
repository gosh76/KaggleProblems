import numpy as np
import pandas as pd
import csv
import lightgbm as lgb
import random
train = pd.read_csv('~/Zt.csv')
props = pd.read_csv('~/Zp.csv',low_memory=False)
#train
t1 = train.merge(props,on='parcelid')
t2 = t1.isnull().sum()
t3 = t2[t2<60000]
t4 = t3.index
t5 = t1[t4]
t5.loc[t5.structuretaxvaluedollarcnt.isnull(),'structuretaxvaluedollarcnt'] = t5.loc[t5.structuretaxvaluedollarcnt.isnull(),'taxvaluedollarcnt'] - \
t5.loc[t5.structuretaxvaluedollarcnt.isnull(),'landtaxvaluedollarcnt']
target = t5.logerror
t5 = t5.drop(['parcelid','transactiondate','logerror','propertycountylandusecode','propertyzoningdesc'],axis=1)
t5 = t5.fillna(t5.median())
trcols = t5.columns
del t1,t2,t3,t4,train
#model
d_train = lgb.Dataset(t5,label=target)
del t5,target
params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'
params['sub_feature'] = 0.27
params['bagging_fraction'] = 0.85
params['bagging_freq'] = 40
params['num_leaves'] = 512
params['min_data'] = 500
params['min_hessian'] = 0.05
params['verbose'] = 1
params['feature_fraction_seed'] = 2
params['bagging_seed'] = 3
np.random.seed(0)
random.seed(0)
model = lgb.train(params, d_train, 400)

#test
ids = list(props.parcelid)
props.loc[props.structuretaxvaluedollarcnt.isnull(),'structuretaxvaluedollarcnt'] = props.loc[props.structuretaxvaluedollarcnt.isnull(),'taxvaluedollarcnt']\
- props.loc[props.structuretaxvaluedollarcnt.isnull(),'landtaxvaluedollarcnt']
props = props[trcols]
props = props.fillna(props.median())
proper = np.array_split(props,100)

#prediction
pro = []
for p in proper:
    pro.append(model.predict(p))

pred0 = np.concatenate(pro)
del props,pro
pred1 = [float('%.4f'%(pred0[i])) for i in range(len(pred0))]
submit1 = pd.DataFrame(ids,columns=['ParcelId'])
submit1['201610'] = pred1
submit1['201611'] = pred1
submit1['201612'] = pred1
submit1['201710'] = pred1
submit1['201711'] = pred1
submit1['201712'] = pred1
submit1.to_csv('Zillow9.csv',index=False)

