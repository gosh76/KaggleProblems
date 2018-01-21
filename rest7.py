import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import GradientBoostingRegressor
pd.set_option('display.max_columns', 380)
pd.set_option('display.width', 140)
pd.set_option('display.max_rows', 380)
ss = pd.read_csv("~/restaurant/sample_submission.csv")
ar = pd.read_csv("~/restaurant/air_reserve.csv")
ast = pd.read_csv("~/restaurant/air_store_info.csv")
av = pd.read_csv("~/restaurant/air_visit_data.csv")
hr = pd.read_csv("~/restaurant/hpg_reserve.csv")
hst = pd.read_csv("~/restaurant/hpg_store_info.csv")
di = pd.read_csv("~/restaurant/date_info.csv")
sir = pd.read_csv("~/restaurant/store_id_relation.csv")
ar['visit_date'] = [str(ar.visit_datetime[i])[:10] for i in range(len(ar))]
ar['visitors'] = ar.reserve_visitors
ar = ar.drop(['visit_datetime','reserve_datetime','reserve_visitors'],axis=1)
s1 = ar.groupby(['air_store_id','visit_date']).sum()
s2 = s1.reset_index()
D5 = dict(zip(sir.hpg_store_id,sir.air_store_id))
list1 = list(D5.keys())
h2 = hr.loc[hr.hpg_store_id.isin(list1),:]
h2['hpg_store_id'] = h2.hpg_store_id.map(lambda x:D5[x])
h2['air_store_id'] = h2.hpg_store_id
h3 = h2.reset_index()
h3 = h3.drop(['index'],axis=1)
h3['visit_date'] = [str(h3.visit_datetime[i])[:10] for i in range(len(h3))]
h3['visitors'] = h3.reserve_visitors
h3 = h3.drop(['hpg_store_id','visit_datetime','reserve_datetime','reserve_visitors'],axis=1)
f1 = h3.groupby(['air_store_id','visit_date']).sum()
f2 = f1.reset_index()
c1 = pd.concat([av,s2,f2])
c2 = c1.groupby(['air_store_id','visit_date']).sum()
c3 = c2.reset_index()
di["visit_date"] = di.calendar_date
di = di.drop(["calendar_date"],axis=1)
av1 = pd.merge(c3,ast,on="air_store_id",how='left')
av2 = pd.merge(av1,di,on="visit_date",how='left')
D4 = dict(zip(av2.day_of_week.unique(),range(len(av2.day_of_week.unique()))))
av2.day_of_week = av2.day_of_week.map(lambda x:D4[x])
av2['y'] = [str(av2.visit_date[i])[0:4] for i in range(len(av2))]
av2['m'] = [str(av2.visit_date[i])[5:7] for i in range(len(av2))]
av2['d'] = [str(av2.visit_date[i])[8:] for i in range(len(av2))]
ss['air_store_id'] = [ss.id[i][0:20] for i in range(len(ss))]
ss1 = pd.merge(ss,ast,on="air_store_id",how='left')
ss1['visit_date'] = [ss1.id[i][21:] for i in range(len(ss1))]
ss2 = pd.merge(ss1,di,on="visit_date",how='left')
ss2['y'] = [str(ss2.visit_date[i])[0:4] for i in range(len(ss2))]
ss2['m'] = [str(ss2.visit_date[i])[5:7] for i in range(len(ss2))]
ss2['d'] = [str(ss2.visit_date[i])[8:] for i in range(len(ss2))]
ss2.day_of_week = ss2.day_of_week.map(lambda x:D4[x])
u1 = list(ss.air_store_id.unique())
soln = []
n = 0
for u in u1:
    z = av2.loc[av2.air_store_id==u,:]
    t = z.visitors
    z = z.drop(['air_store_id','visit_date','visitors','air_genre_name','air_area_name'],axis=1)
    tra = np.array(z,order='C',copy=False)
    tar = np.array(t,order='C',copy=False)
    model = GradientBoostingRegressor(n_estimators=50,random_state=29)
    model = model.fit(tra,tar)
    p = ss2.loc[ss2.air_store_id==u,:]
    v1 = p.id
    p = p.drop(['id','air_store_id','visit_date','visitors','air_genre_name','air_area_name'],axis=1)
    tes = np.array(p,order='C',copy=False)
    pred = model.predict(tes)
    pred1 = [round(pred[i]) for i in range(len(pred))]
    sub1 = pd.DataFrame(v1,columns=['id'])
    sub1['visitors'] = pred1
    soln.append(sub1)
    n = n + 1
    print("n = ",n)

soln1 = pd.concat(soln)
soln1.loc[soln1.visitors<0,'visitors'] = 0
soln1.to_csv('rest7.csv',index=False)


