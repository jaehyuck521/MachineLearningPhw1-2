import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Normalizer,StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings(action='ignore')
# preprocessing
df=pd.read_excel("breast-cancer-wisconsin.xlsx")
print(df.info())
print(df.head())
df.replace('?',np.nan,inplace=True) # some data column has invalid value. So we replace ? to nan
print(df.info())
df['feature6'].fillna(method='ffill',inplace=True) # we fill the nan data with ffill
print(df.info()) # confirm the info
x=df.drop(['target'], axis=1)
y=df['target'].values.reshape(-1,1)
# feature1~9 will be scaled and target will be encoded.
scaler1=StandardScaler()
scaler2=MinMaxScaler()
scaler3=MaxAbsScaler()
scaler4=RobustScaler()
scaler5=Normalizer()
encoder1=LabelEncoder()
encoder2=OrdinalEncoder()
# define scaler and encoder

# function that execute decision tree with Gini index combinations and return the best score and parameter
def DecisionTreeGini(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=2000, shuffle=True)
    model=DecisionTreeRegressor(criterion='mse');
    param={
        'max_depth':range(1,10),
        'min_samples_split':range(2,10),
        'min_samples_leaf':range(1,5),
        'splitter': ['best', 'random']
    }
    # parameters
    gs=GridSearchCV(model,param_grid=param, cv=5, refit=True)
    gs.fit(x_train,y_train)
    be=gs.best_estimator_ # get the best estimator.
    score1=be.score(x_test,y_test) # get the score
    wholelist = []
    wholelist.append(score1)
    wholelist.append(gs.best_params_)
    return wholelist # return the best score and best parameters.

# function that execute decision tree with Entropy combinations and return the best score and parameter
def DecisionTreeEntropy(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=2000, shuffle=True)
    model = DecisionTreeRegressor(criterion='mae');
    param = {
        'max_depth': range(1, 10),
        'min_samples_split': range(2, 10),
        'min_samples_leaf': range(1, 5),
        'splitter': ['best','random']
    }
    # parameters
    gs = GridSearchCV(model, param_grid=param, cv=5, refit=True)
    gs.fit(x_train, y_train)
    be = gs.best_estimator_ # get the best estimator.
    score1 = be.score(x_test, y_test)
    wholelist = []
    wholelist.append(score1)
    wholelist.append(gs.best_params_)
    return wholelist # return the best score and best parameters.


# function that execute Support vector machine combinations and return the best score and parameter
def SVM(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=2000, shuffle=True)
    model=SVC()
    param={
        'C': [0.1, 1.0, 10.0],
        'kernel':['linear','poly','rbf','sigmoid'],
        'degree':range(1,5),
        'gamma':['scale','auto'],
        'shrinking':[True, False],
        'tol': [0.0001, 0.001, 0.001, 0.01],
        'class_weight':['dict','balanced'],
        'max_iter': [200, 400, 600, 800, 1000],
        'decision_function_shape':['ovo','ovr']
    }
    # parameters
    gs = GridSearchCV(model, param_grid=param, cv=5, refit=True)
    gs.fit(x_train, y_train)
    be = gs.best_estimator_ # get the best estimator
    score1 = be.score(x_test, y_test)
    wholelist = []
    wholelist.append(score1)
    wholelist.append(gs.best_params_)
    return wholelist # return the best score and best parameters


# function that execute Logistic regression combinations and return the best score and parameter
def Logistic(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=2000, shuffle=True)
    model = LogisticRegression()
    param = {
        'penalty': ['l2',  'none','l1'],
        'solver': ['lbfgs', 'saga','newton-cg','sag'],
        'max_iter': [200, 400, 600,800,1000],
        'dual': [False, True],
        'tol': [0.0001, 0.001, 0.001, 0.01],
        'C': [0.1, 1.0, 10.0]
    }
    # parameters
    gs = GridSearchCV(model, param_grid=param, cv=5, refit=True)
    gs.fit(x_train, y_train)
    be = gs.best_estimator_ # get the best estimator
    score1 = be.score(x_test, y_test)
    wholelist = []
    wholelist.append(score1)
    wholelist.append(gs.best_params_)
    return wholelist # return the best score and best parameters

# the function that execute many of the combination scaling, encoding, many model and parameters and get the score and
# best parameters so return that value. I judged feature values are numerical value. So it needs to be scaled.
# The target values are categorical so. i apply encoding to it.
# First we execute 4 model without scaling and encoding and get the best model
# and try to execute 10 combination with best model
# and return the model and best combination of scale and encoding
def bestOfbest(x,y):
    x1 = scaler1.fit_transform(x)
    x2 = scaler2.fit_transform(x)
    x3 = scaler3.fit_transform(x)
    x4 = scaler4.fit_transform(x)
    x5 = scaler5.fit_transform(x)
    y1 = encoder1.fit_transform(y)
    y2 = encoder2.fit_transform(y)
    # scaling and encoding: feature and target.
    selectModel={} # for the variable insert the model's best score and parameters.
    for i in range(1,5):
        if i==1:
            selectModel[i]=DecisionTreeEntropy(x,y)
        if i==2:
            selectModel[i]=DecisionTreeGini(x,y)
        if i==3:
            selectModel[i]=SVM(x,y)
        if i==4:
            selectModel[i]=Logistic(x,y)
    # get the best score model and store the index
    # index 1: DecsionTree Entrophy index 2: DecisionTree Gini index 3: SVM index 4: Logistic regression
    max = 0;
    index = 0
    for i in selectModel.keys():
        temp=selectModel[i]
        temp2=temp[0]
        if max<temp2:
            max=temp2
            for j in range (1,5):
                if selectModel[j][0]==max:
                    index=j

    dtList = {}
    # using index we save the best model's name
    # and execute 10 combination of scaling and encoding. and save them to dictionary
    if index==1:
        OptimalModel="Decision Tree with Entropy"
        for k in range(1,11):
            if k==1:
                dtList[k]= DecisionTreeEntropy(x1,y1)
            if k == 2:
                dtList[k] = DecisionTreeEntropy(x2, y1)
            if k == 3:
                dtList[k] = DecisionTreeEntropy(x3, y1)
            if k == 4:
                dtList[k] = DecisionTreeEntropy(x4, y1)
            if k == 5:
                dtList[k] = DecisionTreeEntropy(x5, y1)
            if k == 6:
                dtList[k] = DecisionTreeEntropy(x1, y2)
            if k == 7:
                dtList[k] = DecisionTreeEntropy(x2, y2)
            if k == 8:
                dtList[k] = DecisionTreeEntropy(x3, y2)
            if k == 9:
                dtList[k] = DecisionTreeEntropy(x4, y2)
            if k == 10:
                dtList[k] = DecisionTreeEntropy(x5, y2)
        index2=0
        max2=0
        for key in dtList.keys():
            temp3 = dtList[key]
            temp4 = temp3[0]
            if max2 < temp4:
                max2 = temp4
                for j in range(1, 11):
                    if dtList[j][0] == max2:
                        index2 = j
    # get the best score combination by index
    # using index we save the best model's name
    # and execute 10 combination of scaling and encoding. and save them to dictionary
    if index==2:
        OptimalModel = "Decision Tree with Gini"
        for k in range(1, 11):
            if k == 1:
                dtList[k] = DecisionTreeGini(x1, y1)
            if k == 2:
                dtList[k] = DecisionTreeGini(x2, y1)
            if k == 3:
                dtList[k] = DecisionTreeGini(x3, y1)
            if k == 4:
                dtList[k] = DecisionTreeGini(x4, y1)
            if k == 5:
                dtList[k] = DecisionTreeGini(x5, y1)
            if k == 6:
                dtList[k] = DecisionTreeGini(x1, y2)
            if k == 7:
                dtList[k] = DecisionTreeGini(x2, y2)
            if k == 8:
                dtList[k] = DecisionTreeGini(x3, y2)
            if k == 9:
                dtList[k] = DecisionTreeGini(x4, y2)
            if k == 10:
                dtList[k] = DecisionTreeGini(x5, y2)

        index2 = 0
        max2 = 0
        for key in dtList.keys():
            temp3 = dtList[key]
            temp4 = temp3[0]
            if max2 < temp4:
                max2 = temp4
                for j in range(1, 11):
                    if dtList[j][0] == max2:
                        index2 = j
    # using index we save the best model's name
    # and execute 10 combination of scaling and encoding. and save them to dictionary
    if index==3:
        OptimalModel = "Support Vector machine"
        for k in range(1, 11):
            if k == 1:
                dtList[k] = SVM(x1, y1)
            if k == 2:
                dtList[k] = SVM(x2, y1)
            if k == 3:
                dtList[k] = SVM(x3, y1)
            if k == 4:
                dtList[k] = SVM(x4, y1)
            if k == 5:
                dtList[k] = SVM(x5, y1)
            if k == 6:
                dtList[k] = SVM(x1, y2)
            if k == 7:
                dtList[k] = SVM(x2, y2)
            if k == 8:
                dtList[k] = SVM(x3, y2)
            if k == 9:
                dtList[k] = SVM(x4, y2)
            if k == 10:
                dtList[k] = SVM(x5, y2)

        index2 = 0
        max2 = 0
        for key in dtList.keys():
            temp3 = dtList[key]
            temp4 = temp3[0]
            if max2 < temp4:
                max2 = temp4
                for j in range(1, 11):
                    if dtList[j][0] == max2:
                        index2 = j


    # using index we save the best model's name
    # and execute 10 combination of scaling and encoding. and save them to dictionary
    if index==4:
        OptimalModel="Logistic regression"
        for k in range(1, 11):
            if k == 1:
                dtList[k] = Logistic(x1, y1)
            if k == 2:
                dtList[k] = Logistic(x2, y1)
            if k == 3:
                dtList[k] = Logistic(x3, y1)
            if k == 4:
                dtList[k] = Logistic(x4, y1)
            if k == 5:
                dtList[k] = Logistic(x5, y1)
            if k == 6:
                dtList[k] = Logistic(x1, y2)
            if k == 7:
                dtList[k] = Logistic(x2, y2)
            if k == 8:
                dtList[k] = Logistic(x3, y2)
            if k == 9:
                dtList[k] = Logistic(x4, y2)
            if k == 10:
                dtList[k] = Logistic(x5, y2)

        index2 = 0
        max2 = 0

        for key in dtList.keys():
            temp3 = dtList[key]
            temp4 = temp3[0]
            if max2 < temp4:
                max2 = temp4
                for j in range(1, 11):
                    if dtList[j][0] == max2:
                        index2 = j

        # when finding best combination of scaling and encoding we put the best combination to index2
        # using index 2 we can get the combination name.
        if index2 == 1:
           BestComb="Standard Scaler & Label Encoder"
        if index2 == 2:
            BestComb = "MinMax Scaler & Label Encoder"
        if index2 == 3:
            BestComb = "MaxAbs Scaler & Label Encoder"
        if index2 == 4:
            BestComb = "Robust Scaler & Label Encoder"
        if index2 == 5:
            BestComb = "Normalizer & Label Encoder"
        if index2 == 6:
            BestComb = "Standard Scaler & Ordinal Encoder"
        if index2 == 7:
            BestComb = "MinMax Scaler & Ordinal Encoder"
        if index2 == 8:
            BestComb = "MaxAbs Scaler & Ordinal Encoder"
        if index2 == 9:
            BestComb = "Robust Scaler & Ordinal Encoder"
        if index2 == 10:
            BestComb = "Normalizer & Ordinal Encoder"

    answer={}
    answer[OptimalModel]=dtList[index2]
    answer['Best Combination']=BestComb
    # we put the best model's name and score and parameters and best combination to dictionary
    return answer
# print it
# showing model performance without scaling and encdoing data
print("Decsion Tree with Gini")
print(DecisionTreeGini(x,y))
print("Decsion Tree with Entrophy")
print(DecisionTreeEntropy(x,y))
print("SVM")
print(SVM(x,y))
print("Logistic Regression")
print(Logistic(x,y))

# print the best combination getting all execution
print("The best condition combination")
print(bestOfbest(x,y))









