# Calculate information value
def calculate_woe_iv(dataset, feature, target):
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })
        
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    
    dset = dset.sort_values(by='WoE')
    
    return dset, iv

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df =pd.read_csv(r'G:\bhanu\data\titanic\train.csv')
print(df.head())
print(df.shape)

#%%
features = ['SibSp']
targets = ['Survived']
#%%
calc_iv(df,i,targets,pr=0)
#%%
print(df.drop(['Survived'],axis=1).columns.tolist())
#%%
for col in df.columns:
    if col == 'Survived': continue
    else:
        print('WoE and IV for column: {}'.format(col))
        dg, iv = calculate_woe_iv(df, col, 'Survived')
        #print(dg)
        print('IV score: {:.2f}'.format(iv))
        print('\n')
#%% 
Embarked_df, Embarked_iv =  calculate_woe_iv(df,'Embarked','Survived')
print(Embarked_df)
print(Embarked_iv)
#%%
Pclass_df, Pclass_iv =  calculate_woe_iv(df,'Pclass','Survived')
print(Pclass_df)
print(Pclass_iv)
#%%
Sex_df, Sex_iv =  calculate_woe_iv(df,'Sex','Survived')
print(Sex_df)
print(Sex_iv)
#%%
SibSp_df, SibSp_iv =  calculate_woe_iv(df,'SibSp','Survived')
print(SibSp_df)
print(SibSp_iv)

#%%
def coarse_classer(df, indexloc_1, indexloc_2):
    mean_val = pd.DataFrame(np.mean(pd.DataFrame([df.iloc[indexloc_1], df.iloc[indexloc_2]]))).T
    original = df.drop([indexloc_1, indexloc_2])
    
    coarsed_df = pd.concat([original, mean_val])
    coarsed_df = coarsed_df.sort_values(by='WoE', ascending=False).reset_index(drop=True)
    
    return coarsed_df


SibSp_df1 = coarse_classer(SibSp_df, 5, 6)
SibSp_df1

#WoE for 5 and 8 SibSp is similar and when these 2 classes are merged the WoE is increased 
#note 5 and 8 classes has index values as 5 and 6 as used in code above.
#%%
#now remap the original dataset to replace the old 5 and 8 classes to something new
df['SibSp'].replace({ 5 : 5 , 8 : 8 }, inplace=True)
#chi-squared test:
#%%
#Checking if two categorical variables are independent can be done with 
#Chi-Squared test of independence.
#Null hypothesis: they are independent
#Alternative hypothesis: that they are correlated in some way
from scipy.stats import chi2_contingency
column_list = df.select_dtypes(exclude='float64').columns.tolist()
column_list.remove('Survived')
for i in column_list:
    cross_tab = pd.crosstab(df['Survived'],df[i])
    chi2,p,dof,expected=chi2_contingency(cross_tab.values)
    print('Survived vs {0} p Values is {1:f}'.format(i,p))
    
