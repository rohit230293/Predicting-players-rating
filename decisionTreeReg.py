# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeRegressor

def decisionTreeReg(X_train,X_test,y_train,y_test):
    regressor_desisionTree=DecisionTreeRegressor()
    regressor_desisionTree.fit(X_train,y_train)
    # Predicting the Test set results
    #y_pred = regressor_desisionTree.predict(X_test)
    return(regressor_desisionTree.score(X_test,y_test))
