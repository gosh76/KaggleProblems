import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import GradientBoostingRegressor
pd.set_option('display.max_columns', 380)
pd.set_option('display.width', 140)
pd.set_option('display.max_rows', 380)

#train
trn = pd.read_csv("~/Conductor/train.csv")
tst = pd.read_csv("~/Conductor/test.csv")
target1 = trn.formation_energy_ev_natom
trn1 = trn.drop(['id','formation_energy_ev_natom','bandgap_energy_ev'],axis=1)
target2 = trn.bandgap_energy_ev
tra = np.array(trn1,order='C',copy=False)
tar1 = np.array(target1,order='C',copy=False)
tar2 = np.array(target2,order='C',copy=False)

#test
ids = tst.id
tst1 = tst.drop(['id'],axis=1)
tes = np.array(tst1,order='C',copy=False)

#model
model1 = GradientBoostingRegressor(n_estimators=100,verbose=2,random_state=29)
model1 = model1.fit(tra,tar1)
pred1 = model1.predict(tes)
model2 = GradientBoostingRegressor(n_estimators=100,verbose=2,random_state=29)
model2 = model2.fit(tra,tar2)
pred2 = model2.predict(tes)

sub1 = pd.DataFrame(ids,columns=['id'])
sub1['formation_energy_ev_natom'] = pred1
sub1['bandgap_energy_ev'] = pred2
sub1.loc[sub1.formation_energy_ev_natom<0,'formation_energy_ev_natom'] = 0
sub1.to_csv("cond1.csv",index=False)
