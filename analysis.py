import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from models import *
%matplotlib inline

#define the three values for the mod        
option1 = 0
option2 = 1
option3 = 2

#Case 1: Testing variance change through lower item change
#run the simulation and save results in a dataframe
n_reps = 1000
counter = 0
results = pd.DataFrame(columns=['p_look_high', 'RT', 'choice','model'])
for n_mods,mod in enumerate([option1, option2, option3]):
    item_hi = 10
    item_low = 10-mod
    item_ref = 20
    m1 = model_1(item_hi, item_low, item_ref)
    m2 = model_2(item_hi, item_low, item_ref)
    for z in range(n_reps):
        m1.sim_trial()
        results.loc[counter, 'p_look_high'] = 1-np.mean(m1.att[:len(m1.acc)])
        results.loc[counter, 'RT'] = np.mean(len(m1.acc))
        results.loc[counter,'choice'] = 'bundle' if np.round(m1.acc[-1])==1 else 'reference'  
        results.loc[counter,'model'] = 1
        results.loc[counter,'mod'] = mod
        counter+=1
    for y in range(n_reps):
        m2.sim_trial()
        results.loc[counter, 'p_look_high'] = np.mean(m2.att[:len(m2.acc)])
        results.loc[counter, 'RT'] = np.mean(len(m2.acc))
        results.loc[counter,'choice'] = 'bundle' if np.round(m2.acc[-1])==1 else 'reference' 
        results.loc[counter,'model'] = 2
        results.loc[counter,'mod'] = mod
        counter+=1

#Display results
print('Differences for variance change, average value equal')
print('Model 1 by mod, choice')
z = results.loc[(results['model']==1) & (results['mod']==option1), 'choice'].value_counts()
print('Mod 2: bundle:{}, reference:{}'.format(z['bundle'], z['reference']))
z = results.loc[(results['model']==1) & (results['mod']==option3), 'choice'].value_counts()
print('Mod 6: bundle:{}, reference:{}'.format(z['bundle'], z['reference']))
print('Model 2 by mod, choice')
z = results.loc[(results['model']==2) & (results['mod']==option1), 'choice'].value_counts()
print('Mod 2: bundle:{}, reference:{}'.format(z['bundle'], z['reference']))
z = results.loc[(results['model']==2) & (results['mod']==option3), 'choice'].value_counts()
print('Mod 6: bundle:{}, reference:{}'.format(z['bundle'], z['reference']))

#are there differences in RT?
print('Model 1 by mod, RT')
z = results.loc[(results['model']==1) & (results['mod']==option1), 'RT'].mean()
print('Mod 2: bundle:{}'.format(z))
z = results.loc[(results['model']==1) & (results['mod']==option3), 'RT'].mean()
print('Mod 6: bundle:{}'.format(z))
print('Model 2 by mod, RT')
z = results.loc[(results['model']==2) & (results['mod']==option1), 'RT'].mean()
print('Mod 2: bundle:{}'.format(z))
z = results.loc[(results['model']==2) & (results['mod']==option3), 'RT'].mean()
print('Mod 6: bundle:{}'.format(z))

plt.figure()
sns.distplot(results.loc[(results['model']==1) & (results['mod']==option1), 'RT'], label='mod 2')
sns.distplot(results.loc[(results['model']==1) & (results['mod']==option2), 'RT'], label='mod 5')
sns.distplot(results.loc[(results['model']==1) & (results['mod']==option3), 'RT'], label='mod 8')
plt.legend()
plt.title('RT for Model 1, Increasing Variance')
plt.figure()
sns.distplot(results.loc[(results['model']==2) & (results['mod']==option1), 'RT'], label='mod 2')
sns.distplot(results.loc[(results['model']==2) & (results['mod']==option2), 'RT'], label='mod 5')
sns.distplot(results.loc[(results['model']==2) & (results['mod']==option3), 'RT'], label='mod 8')
plt.legend()
plt.title('RT for Model 2, Increasing Variance')

#are there differences in p_look_high?
print('Model 1, p_look_high')
print('Mod 2: {}'.format(results.loc[(results['model']==1) & (results['mod']==option1), 'p_look_high'].mean()))
print('Mod 6: {}'.format(results.loc[(results['model']==1) & (results['mod']==option3), 'p_look_high'].mean()))
print('Model 2, p_look_high')
print('Mod 2: {}'.format(results.loc[(results['model']==2) & (results['mod']==option1), 'p_look_high'].mean()))
print('Mod 6: {}'.format(results.loc[(results['model']==2) & (results['mod']==option3), 'p_look_high'].mean()))
