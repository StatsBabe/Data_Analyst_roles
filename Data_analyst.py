# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 18:35:36 2020

@author: Owner#
Tanya Reeves
Dim reduction:
We observe that by increasing the number of principal components from 1 to 4, 
the train and test scores improve. This is because with less components, 
there is high bias error in the model, since model is overly simplified. 
As we increase the number of principal components, the bias error will reduce, 
but complexity in the model increases.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso

# remember for regression models, continuous variables (more properly called ordinal features) must be approx N(0,1)
# remember you need to make sure all variables of same scale
# also get rid of highly correlated variables
# good example of what to put on Github
# url = 'https://github.com/bot13956/ML_Model_for_Predicting_Ships_Crew_Size/blob/master/cruise_ship_info.csv'

df = pd.read_csv("ML_Model_for_Predicting_Ships_Crew_Size-master/cruise_ship_info.csv")

df.info()

"""
RangeIndex: 158 entries, 0 to 157
Data columns (total 9 columns):
Ship_name            158 non-null object
Cruise_line          158 non-null object
Age                  158 non-null int64
Tonnage              158 non-null float64
passengers           158 non-null float64
length               158 non-null float64
cabins               158 non-null float64
passenger_density    158 non-null float64
crew                 158 non-null float64
dtypes: float64(6), int64(1), object(2)
"""
df.isnull().sum()
df.isnull().sum().sum()

"""
Out[4]: 
Ship_name            0
Cruise_line          0
Age                  0
Tonnage              0
passengers           0
length               0
cabins               0
passenger_density    0
crew                 0
dtype: int64
"""

df.describe()
"""
              Age     Tonnage  ...  passenger_density        crew
count  158.000000  158.000000  ...         158.000000  158.000000
mean    15.689873   71.284671  ...          39.900949    7.794177
std      7.615691   37.229540  ...           8.639217    3.503487
min      4.000000    2.329000  ...          17.700000    0.590000
25%     10.000000   46.013000  ...          34.570000    5.480000
50%     14.000000   71.899000  ...          39.085000    8.150000
75%     20.000000   90.772500  ...          44.185000    9.990000
max     48.000000  220.000000  ...          71.430000   21.000000
"""

cols = ['Age', 'Tonnage', 'passengers', 'length', 'cabins','passenger_density','crew']

sns.pairplot(df[cols], size=2.0)

sns.distplot(df['Age'],bins=20)
plt.title('probability distribution')
plt.show()

sns.distplot(df['Tonnage'],bins=20)
plt.title('probability distribution')
plt.show()

cols = ['Age', 'Tonnage', 'passengers', 'length', 'cabins','passenger_density','crew']
stdsc = StandardScaler()
X_std = stdsc.fit_transform(df[cols].iloc[:,range(0,7)].values)

cov_mat =np.cov(X_std.T)
plt.figure(figsize=(15,15))
sns.set(font_scale=1.5)
hm = sns.heatmap(cov_mat,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 12},
                 yticklabels=cols,
                 xticklabels=cols)
plt.title('Covariance matrix showing correlation coefficients')
plt.tight_layout()
plt.show()

# "crew" variable correlates strongly with 4 predictor variables: "Tonnage", "passengers", "length, and "cabins"
# therefore
cols_selected = ['Tonnage', 'passengers', 'length', 'cabins','crew']
df[cols_selected].head()
"""
   Tonnage  passengers  length  cabins   crew
0   30.277        6.94    5.94    3.55   3.55
1   30.277        6.94    5.94    3.55   3.55
2   47.262       14.86    7.22    7.43   6.70
3  110.000       29.74    9.53   14.88  19.10
4  101.353       26.42    8.92   13.21  10.00
"""

X = df[cols_selected].iloc[:,0:4].values    # features matrix 
y = df[cols_selected]['crew'].values        # target variable

X.shape
y.shape

df.shape

ohe = OneHotEncoder(categorical_features=[0 ,1])
pd.get_dummies(df[['Ship_name', 'Cruise_line','Age', 'Tonnage', 'passengers', 'length', 'cabins','passenger_density','crew']])

df2=pd.get_dummies(df[['Ship_name', 'Cruise_line','Age', 'Tonnage', 'passengers', 'length', 'cabins','passenger_density','crew']])
df2.head()
"""
   Age  Tonnage  ...  Cruise_line_Star  Cruise_line_Windstar
0    6   30.277  ...                 0                     0
1    6   30.277  ...                 0                     0
2   26   47.262  ...                 0                     0
3   11  110.000  ...                 0                     0
4   17  101.353  ...                 0                     0
"""

plt.scatter(df2['Ship_name_Adventure'],df2['crew'])
plt.xlabel('Ship_name_Adventure')
plt.ylabel('crew')
plt.show()

# The categorical features "Ship_name" and "Cruise_line" will not be used.
# use 4 ordinal (continuous data) features "Tonnage", "passengers", "length, and "cabins".

# remember we already created cols_selected = ['Tonnage', 'passengers', 'length', 'cabins','crew']
X = df[cols_selected].iloc[:,0:4].values     
y = df[cols_selected]['crew']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=0)

slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()
plt.legend(loc='lower right')
plt.show()

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
"""
MSE train: 0.955, test: 0.889
R^2 train: 0.920, test: 0.928

"""

slr.fit(X_train, y_train).intercept_
# -0.7525074496158339

slr.fit(X_train, y_train).coef_
# array([ 0.01902703, -0.15001099,  0.37876395,  0.77613801])

X = df[cols_selected].iloc[:,0:4].values     
y = df[cols_selected]['crew']  

sc_y = StandardScaler()
sc_x = StandardScaler()
y_std = sc_y.fit_transform(y_train[:, np.newaxis]).flatten()

train_score = []
test_score = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=i)
    y_train_std = sc_y.fit_transform(y_train[:, np.newaxis]).flatten()
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=4)),('slr', LinearRegression())])
    pipe_lr.fit(X_train, y_train_std)
    y_train_pred_std=pipe_lr.predict(X_train)
    y_test_pred_std=pipe_lr.predict(X_test)
    y_train_pred=sc_y.inverse_transform(y_train_pred_std)
    y_test_pred=sc_y.inverse_transform(y_test_pred_std)
    train_score = np.append(train_score, r2_score(y_train, y_train_pred))
    test_score = np.append(test_score, r2_score(y_test, y_test_pred))

train_score
# array([0.92028261, 0.91733937, 0.94839385, 0.93899476, 0.90621451,
#       0.91156903, 0.92726066, 0.94000795, 0.93922948, 0.93629554])
test_score
# array([0.92827978, 0.93807946, 0.8741834 , 0.89901199, 0.94781315,
#       0.91880183, 0.91437408, 0.89660876, 0.90427477, 0.90139208])  

print('R2 train: %.3f +/- %.3f' % (np.mean(train_score),np.std(train_score))) 
# R2 train: 0.929 +/- 0.013

print('R2 test: %.3f +/- %.3f' % (np.mean(test_score),np.std(test_score)))
# R2 test: 0.912 +/- 0.021

# PCA

train_score = []
test_score = []
cum_variance = []

for i in range(1,5):
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=0)
    y_train_std = sc_y.fit_transform(y_train[:, np.newaxis]).flatten()
    
    pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=i)),('slr', LinearRegression())])
    pipe_lr.fit(X_train, y_train_std)
    y_train_pred_std=pipe_lr.predict(X_train)
    y_test_pred_std=pipe_lr.predict(X_test)
    y_train_pred=sc_y.inverse_transform(y_train_pred_std)
    y_test_pred=sc_y.inverse_transform(y_test_pred_std)
    train_score = np.append(train_score, r2_score(y_train, y_train_pred))
    test_score = np.append(test_score, r2_score(y_test, y_test_pred))
    cum_variance = np.append(cum_variance, np.sum(pipe_lr.fit(X_train, y_train).named_steps['pca'].explained_variance_ratio_))

train_score
# Out[60]: array([0.90411898, 0.9041488 , 0.90416405, 0.92028261])

test_score
# Out[61]: array([0.89217843, 0.89174896, 0.89159266, 0.92827978])

cum_variance
# Out[62]: array([0.949817  , 0.98322819, 0.99587366, 1.        ])

plt.scatter(cum_variance,train_score, label = 'train_score')
plt.plot(cum_variance, train_score)
plt.scatter(cum_variance,test_score, label = 'test_score')
plt.plot(cum_variance, test_score)
plt.xlabel('cumulative variance')
plt.ylabel('R2_score')
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=0)
y_train_std = sc_y.fit_transform(y_train[:, np.newaxis]).flatten()
X_train_std = sc_x.fit_transform(X_train)
X_test_std = sc_x.transform(X_test)

alpha = np.linspace(0.01,0.4,10)

lasso = Lasso(alpha=0.7)

r2_train=[]
r2_test=[]
norm = []
for i in range(10):
    lasso = Lasso(alpha=alpha[i])
    lasso.fit(X_train_std,y_train_std)
    y_train_std=lasso.predict(X_train_std)
    y_test_std=lasso.predict(X_test_std)
    r2_train=np.append(r2_train,r2_score(y_train,sc_y.inverse_transform(y_train_std)))
    r2_test=np.append(r2_test,r2_score(y_test,sc_y.inverse_transform(y_test_std)))
    norm= np.append(norm,np.linalg.norm(lasso.coef_))
    
lasso = Lasso(alpha=0.7)

plt.scatter(alpha,r2_train,label='r2_train')
plt.plot(alpha,r2_train)
plt.scatter(alpha,r2_test,label='r2_test')
plt.plot(alpha,r2_test)
plt.scatter(alpha,norm,label = 'norm')
plt.plot(alpha,norm)
plt.ylim(-0.1,1)
plt.xlim(0,.43)
plt.xlabel('alpha')
plt.ylabel('R2_score')
plt.legend()
plt.show()
"""
We observe that as the regularization parameter $\alpha$ increases, the norm of the regression coefficients become smaller and smaller. 
This means more regression coefficients are forced to zero, which intend increases bias error 
(over simplification). The best value to balance bias-variance tradeoff is when $\alpha$ is kept low, say $\alpha = 0.1$ or less.
"""
