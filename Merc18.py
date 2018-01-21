import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import GradientBoostingRegressor
pd.set_option('display.max_columns', 380)
pd.set_option('display.width', 140)
pd.set_option('display.max_rows', 380)
trn = pd.read_csv("../input/train.tsv",delimiter="\t")
tst = pd.read_csv("../input/test.tsv",delimiter="\t")
trn.isnull().sum()
tst.isnull().sum()
trn.loc[trn.name.isnull(),'name'] = 'None'
trn.loc[trn.item_condition_id.isnull(),'item_condition_id'] = trn.item_condition_id.mode()[0]
trn.loc[trn.category_name.isnull(),'category_name'] = 'None'
trn.loc[trn.brand_name.isnull(),'brand_name'] = 'None'
trn.loc[trn.price.isnull(),'price'] = trn.price.mode()[0]
trn.loc[trn.shipping.isnull(),'shipping'] = trn.shipping.mode()[0]
trn.loc[trn.item_description.isnull(),'item_description'] = 'None'
tst.loc[tst.name.isnull(),'name'] = 'None'
tst.loc[tst.item_condition_id.isnull(),'item_condition_id'] = tst.item_condition_id.mode()[0]
tst.loc[tst.category_name.isnull(),'category_name'] = 'None'
tst.loc[tst.brand_name.isnull(),'brand_name'] = 'None'
tst.loc[tst.shipping.isnull(),'shipping'] = tst.shipping.mode()[0]
tst.loc[tst.item_description.isnull(),'item_description'] = 'None'
target = trn.price
ids = tst.test_id
trn = trn.drop(['train_id','price'],axis=1)
tst = tst.drop(['test_id'],axis=1)
c1 = pd.concat([trn,tst])
r1 = trn.shape[0]
del trn,tst
c1.name = c1.name.astype('category').cat.codes
c1.category_name = c1.category_name.astype('category').cat.codes
c1.brand_name = c1.brand_name.astype('category').cat.codes
c1.item_description = c1.item_description.astype('category').cat.codes
trn1 = c1[:r1]
tst1 = c1[r1:]
tra = np.array(trn1,order='C',copy=False)
tar = np.array(target,order='C',copy=False)
tes = np.array(tst1,order='C',copy=False)

#model
model = GradientBoostingRegressor(n_estimators=170,min_samples_split=2,max_depth=8,verbose=2,random_state=29)
model = model.fit(tra,tar)
pred = model.predict(tes)
sub1 = pd.DataFrame(ids,columns=['test_id'])
sub1['price'] = pred
sub1.loc[sub1.price<0,'price'] = 0
sub1.to_csv('Merc18.csv',index=False)


