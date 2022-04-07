# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from patsy import dmatrices
import warnings
warnings.filterwarnings(action='ignore')


# Importing the dataset
df = pd.read_csv('FG_LHV_r1.csv')

# Drop Unnecessary Columns
df = df.drop(['Date'], axis=1)

# Target Variable Distribution
Target_feature = list(set(['FG_LHV_Lab']))
Target_feature = np.sort(Target_feature)

for col in Target_feature:
    sns.distplot(df.loc[df[col].notnull(), col])
    plt.title(col)
    plt.show()

# Input Variable Distribution
numerical_feature = list(set(df.columns) - set(['FG_LHV_Lab']))
numerical_feature = np.sort(numerical_feature)

for col in numerical_feature:
    sns.distplot(df.loc[df[col].notnull(), col])
    plt.title(col)
    plt.show()

# Multicollinearity check
from statsmodels.stats.outliers_influence import variance_inflation_factor
y, X = dmatrices('df.iloc[:, -1] ~ df.iloc[:, :-1]', df, return_type = 'dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns 
vif

# Correlation Check
df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(data = df.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='Reds')      

# Drop Unnecessary Columns
df = df.drop(['FG_Press_A'], axis=1)
df = df.drop(['FG_Press_B'], axis=1)
df = df.drop(['FG_Flow_A'], axis=1)
df = df.drop(['FG_Flow_B'], axis=1)
df = df.drop(['HTR_Inlet_Temp'], axis=1)
df = df.drop(['HTR_Outlet_Temp'], axis=1)
df = df.drop(['HTR_Inlet_Temp_r1'], axis=1)
df = df.drop(['HTR_Outlet_Temp_r1_A'], axis=1)
df = df.drop(['HTR_Outlet_Temp_r1_B'], axis=1)
df = df.drop(['HTR_O2_A'], axis=1)
df = df.drop(['FG_Flow_Sum'], axis=1)
df = df.drop(['HTR_Flow'], axis=1)
df = df.drop(['HTR_O2_Sum'], axis=1)
df = df.drop(['HTR_Delta_Temp'], axis=1)
df = df.drop(['FG_LHV'], axis=1)

# Data Information
df.head()
df.info()

# Null check
df.isnull().sum()

# Multicollinearity check
from statsmodels.stats.outliers_influence import variance_inflation_factor
y, X = dmatrices('df.iloc[:, -1] ~ df.iloc[:, :-1]', df, return_type = 'dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns 
vif

# Correlation Check
df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(data = df.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='Reds')         

# Target Variable 중 Null 열 제거
df1 = df.dropna(axis=0)

#---------------------------------------------------------------------------------------------------------

# X, y variable selection              
X = df1.iloc[:, :-1]
y = df1.iloc[:, -1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=100)

plt.rcParams["figure.figsize"] = (18,8)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.grid'] = True
x_values = range(0, 36)
plt.plot(x_values, y_test, marker='o')
plt.legend(['y_test'])
plt.xlabel('Case')
plt.ylabel('F/G LHV (BTU/Nm3)')
plt.title('F/G LHV Prediction for y_test')
plt.show()


#---------------------------------------------------------------------------------------------------------

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept=True, normalize=True)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred_ols = regressor.predict(X_test)
print('Accuracy: %.4f' % r2_score(y_test, y_pred_ols))

# Caculating MSE
mse = np.mean((y_pred_ols - y_test)**2)
print('MSE: %.4f' % mse)

# Checking the magnitude of coefficients
predictors = X_train.columns
coef = Series(regressor.coef_, predictors).sort_values()
coef.plot(kind='bar', title='Model Coefficients')


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

for n in [3,4,5,6,7,8]:
    kfold = KFold(n_splits=n, shuffle=True, random_state=100)
    scores = cross_val_score(regressor, X, y, cv=kfold) # model, train, target, cross validation
    print('cross-val-score \n{}'.format(scores))
    print('cross-val-score.mean \n{:.3f}'.format(scores.mean()))


#---------------------------------------------------------------------------------------------------------

# Fitting the PCA algorithm with our Data
pca = PCA().fit(X_train)

# Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Dataset Explained Variance')
plt.show()

# Applying PCA function on training 
# and testing set of X component 
pca = PCA(n_components = 5)
  
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
  
explained_variance = pca.explained_variance_ratio_ 

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor_pca = LinearRegression()
regressor_pca.fit(X_train_pca, y_train)

# Predicting the Test set results
y_pred_ols_pca = regressor_pca.predict(X_test_pca)
print('Accuracy: %.4f' % r2_score(y_test, y_pred_ols_pca))

# Caculating MSE
mse = np.mean((y_pred_ols_pca - y_test)**2)
print('MSE: %.4f' % mse)

#---------------------------------------------------------------------------------------------------------

# Lasso Regression (Least Absolute Shrinkage Selector Operator)
from sklearn.linear_model import Lasso

lassoReg = Lasso(alpha= 0.6, normalize = True)
lassoReg.fit(X_train, y_train)

pred = lassoReg.predict(X_test)

# Predicting the Test set results
y_pred_lasso = pred
print('Accuracy: %.4f' % r2_score(y_test, y_pred_lasso))

# Caculating MSE
mse = np.mean((y_pred_lasso - y_test)**2)
print('MSE: %.4f' % mse)

# Checking the magnitude of coefficients
predictors = X_train.columns
coef = Series(lassoReg.coef_, predictors).sort_values()
coef.plot(kind='bar', title='Model Coefficients')


#---------------------------------------------------------------------------------------------------------

# Elastic Net Regression
from sklearn.linear_model import ElasticNet

ENReg = ElasticNet(alpha = 0.01, l1_ratio = 0.7, normalize = False)
ENReg.fit(X_train, y_train)

pred = ENReg.predict(X_test)

# Predicting the Test set results
y_pred_EN = pred
print('Accuracy: %.4f' % r2_score(y_test, y_pred_EN))

# Caculating MSE
mse = np.mean((y_pred_EN - y_test)**2)
print('MSE: %.4f' % mse)

# Checking the magnitude of coefficients
predictors = X_train.columns
coef = Series(ENReg.coef_, predictors).sort_values()
coef.plot(kind='bar', title='Model Coefficients')


#---------------------------------------------------------------------------------------------------------

# XGBoost Regression
from xgboost import XGBRegressor
from xgboost import plot_importance

XGB = XGBRegressor(silent=False, 
                   booster='gbtree',
                   min_child_weight = 8, # default = 1, 높을수록 under-fitting
                   max_depth=3, # default = 6, typical 3 ~ 10
                   colsample_bytree = 0.9, # 트리 생성 시 훈련 데이터 변수 샘플링 비율, 보통 0.6 ~ 0.9
                   colsample_bylevel = 0.9, # 트리의 레벨 별로 훈련 데이터 변수 샘플링 비율, 보통 0.6 ~ 0.9
                   subsample = 0.499,
                   objective='reg:linear', 
                   eval_metric = 'rmse',
                   n_estimators=61, 
                   seed=100)

XGB.fit(X_train, y_train)
pred = XGB.predict(X_test)

# Predicting the Test set results
y_pred_XGB = pred
print('Accuracy: %.4f' % r2_score(y_test, y_pred_XGB))

# Caculating MSE
mse = np.mean((y_pred_XGB - y_test)**2)
print('MSE: %.4f' % mse)

# Checking the feature importance
plot_importance(XGB)

#---------------------------------------------------------------------------------------------------------

# 각 Model 별 오차범위 미만 예측률/ 평균 Delta 값/ R2 Scores
ErrorRange = 2200

exactSuccessCount = 0.00
exactFailCount = 0.00
score1 = 0.00
i = 0
for row in y_test:
    if abs(row - y_pred_ols[i]) <= ErrorRange:
        exactSuccessCount = exactSuccessCount + 1
    else:
        exactFailCount = exactFailCount + 1
    i = i + 1
print("-------------------------------------------")
print("Row 개수 = {}".format(len(y_test)))
print("Success 개수 = {:.0f}".format(exactSuccessCount))
print("Fail 개수 = {:.0f}".format(exactFailCount))
score1 = exactSuccessCount/(exactSuccessCount+exactFailCount) * 100
print("y_pred_ols : 오차범위 미만 예측률 = {:.2f} %".format(score1))
i=0
a=0
for row in y_test:
    a = a + abs(y_pred_ols[i] - row)
    i = i + 1
a = a / 36
print("y_pred_ols - y_test 평균 절대값: {:.1f}".format(a))
print('Accuracy of OLS : %.4f' % r2_score(y_test, y_pred_ols))
print("-------------------------------------------")

exactSuccessCount = 0.00
exactFailCount = 0.00
i = 0
for row in y_test:
    if abs(row - y_pred_ols_pca[i]) <= ErrorRange:
        exactSuccessCount = exactSuccessCount + 1
    else:
        exactFailCount = exactFailCount + 1
    i = i + 1
print("Row 개수 = {}".format(len(y_test)))
print("Success 개수 = {:.0f}".format(exactSuccessCount))
print("Fail 개수 = {:.0f}".format(exactFailCount))
score1 = exactSuccessCount/(exactSuccessCount+exactFailCount) * 100
print("y_pred_ols_pca : 오차범위 미만 예측률 = {:.2f} %".format(score1))
i=0
a=0
for row in y_test:
    a = a + abs(y_pred_ols_pca[i] - row)
    i = i + 1
a = a / 36
print("y_pred_ols_pca - y_test 평균 절대값: {:.1f}".format(a))
print('Accuracy of PCA : %.4f' % r2_score(y_test, y_pred_ols_pca))
print("-------------------------------------------")

exactSuccessCount = 0.00
exactFailCount = 0.00
i = 0
for row in y_test:
    if abs(row - y_pred_lasso[i]) <= ErrorRange:
        exactSuccessCount = exactSuccessCount + 1
    else:
        exactFailCount = exactFailCount + 1
    i = i + 1
print("Row 개수 = {}".format(len(y_test)))
print("Success 개수 = {:.0f}".format(exactSuccessCount))
print("Fail 개수 = {:.0f}".format(exactFailCount))
score1 = exactSuccessCount/(exactSuccessCount+exactFailCount) * 100
print("y_pred_lasso : 오차범위 미만 예측률 = {:.2f} %".format(score1))
i=0
a=0
for row in y_test:
    a = a + abs(y_pred_lasso[i] - row)
    i = i + 1
a = a / 36
print("y_pred_lasso - y_test 평균 절대값: {:.1f}".format(a))
print('Accuracy of LASSO : %.4f' % r2_score(y_test, y_pred_lasso))
print("-------------------------------------------")

exactSuccessCount = 0.00
exactFailCount = 0.00
i = 0
for row in y_test:
    if abs(row - y_pred_EN[i]) <= ErrorRange:
        exactSuccessCount = exactSuccessCount + 1
    else:
        exactFailCount = exactFailCount + 1
    i = i + 1
print("Row 개수 = {}".format(len(y_test)))
print("Success 개수 = {:.0f}".format(exactSuccessCount))
print("Fail 개수 = {:.0f}".format(exactFailCount))
score1 = exactSuccessCount/(exactSuccessCount+exactFailCount) * 100
print("y_pred_EN : 오차범위 미만 예측률 = {:.2f} %".format(score1))
i=0
a=0
for row in y_test:
    a = a + abs(y_pred_EN[i] - row)
    i = i + 1
a = a / 36
print("y_pred_EN - y_test 평균 절대값: {:.1f}".format(a))
print('Accuracy of Elastic Net : %.4f' % r2_score(y_test, y_pred_EN))
print("-------------------------------------------")

exactSuccessCount = 0.00
exactFailCount = 0.00
i = 0
for row in y_test:
    if abs(row - y_pred_XGB[i]) <= ErrorRange:
        exactSuccessCount = exactSuccessCount + 1
    else:
        exactFailCount = exactFailCount + 1
    i = i + 1
print("Row 개수 = {}".format(len(y_test)))
print("Success 개수 = {:.0f}".format(exactSuccessCount))
print("Fail 개수 = {:.0f}".format(exactFailCount))
score1 = exactSuccessCount/(exactSuccessCount+exactFailCount) * 100
print("y_pred_XGB : 오차범위 미만 예측률 = {:.2f} %".format(score1))
i=0
a=0
for row in y_test:
    a = a + abs(y_pred_XGB[i] - row)
    i = i + 1
a = a / 36
print("y_pred_XGB - y_test 평균 절대값: {:.1f}".format(a))
print('Accuracy of XGBoost : %.4f' % r2_score(y_test, y_pred_XGB))
print("-------------------------------------------")




# 각 Model 결과 별 Graph 그리기
plt.rcParams["figure.figsize"] = (18,8)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.grid'] = True

x_values = range(0, 36)
plt.plot(x_values, y_test, marker='o')
plt.plot(x_values, y_pred_ols, marker='o')
plt.plot(x_values, y_pred_ols_pca, marker='o')
plt.plot(x_values, y_pred_lasso, marker='o')
plt.plot(x_values, y_pred_EN, marker='o')
plt.plot(x_values, y_pred_XGB, marker='o')
plt.legend(['y_test', 'y_pred_ols','y_pred_ols_pca', 'y_pred_lasso', 'y_pred_EN', 'y_pred_XGB'])
plt.xlabel('Case')
plt.ylabel('F/G LHV (BTU/Nm3)')
plt.title('F/G LHV Prediction for 5 models')
plt.show()

plt.rcParams["figure.figsize"] = (18,8)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.grid'] = True
x_values = range(0, 36)
plt.plot(x_values, y_test, marker='o')
plt.plot(x_values, y_pred_XGB, marker='o')
plt.legend(['y_test', 'y_pred_XGB'])
plt.xlabel('Case')
plt.ylabel('F/G LHV (BTU/Nm3)')
plt.title('F/G LHV Prediction for XGBoost model')
plt.show()