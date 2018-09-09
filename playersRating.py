import numpy as np

# importing the database
import readdata
df = readdata.readdata('database.sqlite')
print(df.head())

#Shuffle the rows of df so we get a distributed sample when we display top few rows
df = df.reindex(np.random.permutation(df.index))

# Defining the X anf y parameters
X = df.iloc[:,1:]   # 'other than overall rating, i.e. column 1'
y = df.iloc[:,0]    #  Need to predict 'Overall rating' 

# Identifying the unique values for preferred_foot colomns
print("Unique values in preferred_foot feature are: ", end=" ")
print(df.preferred_foot.unique())
print("Lets apply one Hot encoding to preferred_foot feature")

# Applying One hot encoding on only catagorical columns, i.e. preferred_foot
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X.preferred_foot = label_encoder.fit_transform(X.preferred_foot)

# Splitting the train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.10,random_state=0)

# Find the most significant features using backward elemination 
print('-'*50)
import backwardEle
backwardEle.backwardEle(X_train, y_train)

print('-'*50)
# Applying Linear regression
import linearReg
linearReg_score = linearReg.linearReg(X_train,X_test,y_train,y_test)
print("Pridicting based on Simple Linear regression we get the score as : {:.5}%" \
                                      .format(linearReg_score*100))
print('-'*50)

# Applying the XGBoost
import xgb
reg_xg_score = xgb.xgb(X_train,X_test,y_train,y_test)
print("Pridicting after applying XGBoost we get the score as : {:.5}%" \
                                      .format(reg_xg_score*100))
print('-'*50)

# Decision Tree

# Fitting Decision Tree Regression to the Training set
import decisionTreeReg
reg_decisionTreeReg_score = decisionTreeReg.decisionTreeReg(X_train,X_test,y_train,y_test)
print("Pridicting after applying Decision Tree Regression we get the score as : {:.5}%" \
                                      .format(reg_decisionTreeReg_score*100))
print('-'*50)


######################### RandomForestRegressor ##########################
from sklearn.ensemble import RandomForestRegressor
regr_rf = RandomForestRegressor(max_depth=30, random_state=2)
regr_rf.fit(X_train, y_train)
y_rf = regr_rf.predict(X_test)
regr_rf.score(X_test,y_test)

################
# Try different numbers of n_estimators - this will take a minute or so
import randForestReg
reg_randForestReg_score = randForestReg.randForestReg(X_train,X_test,y_train,y_test)
print("Pridicting after applying Random Forest Regression with best n_estimator we get the score as : {:.5}%" \
                                      .format(reg_randForestReg_score*100))
print('-'*50)
