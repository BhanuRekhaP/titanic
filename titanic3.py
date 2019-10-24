#%%
#Reading data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df =pd.read_csv(r'G:\bhanu\data\titanic\train.csv')
X_test  = pd.read_csv(r'G:\bhanu\data\titanic\test.csv')

print(df.columns.tolist())
#%%
print(df.info())
#%%
print(df.isnull().sum())
#%%

df[['Sex','Embarked']]= df[['Sex','Embarked']].astype('object')

#%%
# drop unnecessary features
df.drop(['Cabin','PassengerId','Ticket','Name'],axis=1,inplace=True)
#%%
#changing the string variables to numeric to run a knnimputer

df.head()
def func1(x):
    if x == 'male':
        return 1
    if x == 'female':
        return 0
        
df['Sex']= df['Sex'].apply(func1)
#%%
def func2(x):
    if x == 'S':
        return 0
    elif x == 'C':
        return 1
    elif x == 'Q':
        return 2
df['Embarked'] = df['Embarked'].apply(func2)


#%%
print(df['Embarked'].value_counts())
print(df['Sex'].value_counts())
#%%
#droping nulls from embarked alone
df.dropna(subset=['Embarked'],axis=0,inplace=True)

#%%
print(df.shape)
#%%
from missingpy import KNNImputer
imputer = KNNImputer(n_neighbors=2, weights="uniform")

X_imputed = imputer.fit_transform(df)
df['Age_imputed']=X_imputed[:,3].round()
print(df[df['Age'].isnull()])
#%%
df.drop(['Age'],axis=1,inplace=True)
sns.boxplot(df['Age_imputed'])
plt.show()
#%%
print(df['Age_imputed'].describe())
#%%
#age_imputed to bins
def func3(x):
    if x < 21:
        return 0
    elif x>= 21 and x <38:
        return 1
    elif x>=38:
        return 2

print(df['Age_imputed'].apply(func3).value_counts())
#%%
df['Age_groups'] = df['Age_imputed'].apply(func3)
df.drop(['Age_imputed'],axis=1,inplace=True)

#%%
sns.boxplot(df['Fare'])
plt.show()
#%%
print(df['Fare'].describe())
#%%
#df['Fare_groups'] = pd.qcut(df.loc[:,'Fare'], [0, .25, .5, .75, 1], labels = ['0 - 7','7 - 14','14 - 31','31 - 512'])
#df.drop(['Fare'],axis=1,inplace=True)
#%%
df[df.drop(['Fare'],axis=1).columns.tolist()]=df[df.drop(['Fare'],axis=1).columns.tolist()].astype('category')
print(df.info())
#%%
#working on test data
X_test[['Sex','Embarked']]= X_test[['Sex','Embarked']].astype('object')
X_test.drop(['Cabin','Ticket','Name'],axis=1,inplace=True)
X_test['Sex']= X_test['Sex'].apply(func1)
X_test['Embarked'] = X_test['Embarked'].apply(func2)
X_test.dropna(subset=['Embarked'],axis=0,inplace=True)
pid = pd.DataFrame(X_test['PassengerId'])
X_test.drop(['PassengerId'],axis=1,inplace=True)
#%%
imputer = KNNImputer(n_neighbors=2, weights="uniform")

X_imputed = imputer.fit_transform(X_test)
X_test['Age_imputed']=X_imputed[:,2].round()
print(X_test[X_test['Age'].isnull()])
#%%
X_test.drop(['Age'],axis=1,inplace=True)
sns.boxplot(X_test['Age_imputed'])
plt.show()
#%%
X_test['Age_groups'] = X_test['Age_imputed'].apply(func3)
X_test.drop(['Age_imputed'],axis=1,inplace=True)
#%%
sns.boxplot(X_test['Fare'])
plt.show()
#%%
print(X_test['Fare'].describe())
#%%
#X_test['Fare_groups'] = pd.qcut(X_test.loc[:,'Fare'], [0, .25, .5, .75, 1], labels = ['0 - 7','7 - 14','14 - 31','31 - 512'])
#X_test.drop(['Fare'],axis=1,inplace=True)
#%%
X_test[X_test.drop(['Fare'],axis=1).columns.tolist()]=X_test[X_test.drop(['Fare'],axis=1).columns.tolist()].astype('category')
print(X_test.info())
#%%
#dummies
#%%
dum_X_train = pd.get_dummies(data = df.drop(['Survived'],axis=1), prefix = None, prefix_sep='_',drop_first =True, dtype='int8')
dum_X_test  = pd.get_dummies(data = X_test, prefix = None, prefix_sep='_',drop_first =True, dtype='int8')
y_train     = df['Survived']

print(dum_X_train.shape)
print(dum_X_test.shape)

print(dum_X_train.columns.tolist())
print(dum_X_test.columns.tolist())

print(dum_X_test.Parch_9.value_counts())

dum_X_test.drop(['Parch_9'],axis=1,inplace=True)

#%%
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
#
for col in df.columns:
    if col == 'Survived': continue
    else:
        print('WoE and IV for column: {}'.format(col))
        dg, iv = calculate_woe_iv(df, col, 'Survived')
        #print(dg)
        print('IV score: {:.2f}'.format(iv))
        print('\n')

#%%
#train test split
from sklearn.model_selection import train_test_split
Xf_train, Xf_test, yf_train, yf_test = train_test_split(dum_X_train, 
                                                    y_train, 
                                                    test_size=0.1, 
                                                    random_state=1)

#%%
#model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Create a dictionary containing all the candidate values of the parameters

parameter_grid = dict(n_estimators=list(range(1, 500, 100)),
                      criterion=['gini','entropy'],
                      max_features=list(range(1, 20, 2)),
                      max_depth= [None] + list(range(5, 25, 1)))

# Creata a random forest object
random_forest = RandomForestClassifier(random_state=0, n_jobs=-1)

# Create a gridsearch object with 5-fold cross validation, and uses all cores (n_jobs=-1)
clf = GridSearchCV(estimator=random_forest, param_grid=parameter_grid, cv=2, verbose=1, n_jobs=-1)

clf.fit(Xf_train, yf_train)
#%%
# Predict who survived in the train dataset
predictions = clf.predict(Xf_test)
#%%
from sklearn.metrics import accuracy_score
print(accuracy_score(yf_test,predictions))
#%%
# Predict who survived in the test dataset
predictions = pd.DataFrame(clf.predict(dum_X_test),columns=['predictions'])
#%%
sub_df = pd.concat([pid,predictions],axis=1)
#sub_df.rename(columns={'PassengerId':'pid',0:'predictions'},inplace=True)

sub_df.to_csv(r'G:\bhanu\data\titanic\titanic_passenger_predictions.csv',index=False) 
