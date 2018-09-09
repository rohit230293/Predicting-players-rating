from sklearn.linear_model import LinearRegression

def linearReg(X_train,X_test,y_train,y_test):
    # Applying Linear regression
    # follow the usual sklearn pattern: import, instantiate, fit
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    # print intercept and coefficients
    #print(lm.intercept_)
    #print(lm.coef_)
    
    # Pridicting the values using liniar regression
    lm.predict(X_test)
      
    return(lm.score(X_test,y_test))
