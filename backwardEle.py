import numpy as np

def backwardEle(X_train, y_train):
    ######################### Backward elemination ##################################
    
    # Create an X_opt matrix, which initially have all the columns from matrix X
    # based on P values of columns we will eleminate them one by one
    
    X_opt = X_train[::]
    
    # Adding b0 in the multi linear regression eq i.e. y = b0 + b1X1 + b2X2 + ...
    # Since sm.OLS lib does not include b0 and user has to take care of this
    # See the parameters for OLS for more details
    import statsmodels.formula.api as sm
    X_opt = np.append(arr=np.ones((X_opt.shape[0], 1)), values=X_opt, axis=1)
    
    # Step 1: Select the significance lavel to stay in the model (tyically SL = 0.005)
    
    # Step 2: Fit the full model with possible pridiction 
    # Create a new object for OLS class i.e. Ordenary least sq
    regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
         
    # Step 3: Consider the predictor with the highest P-value
    # if P > SL, go to step 4, else Done
    print('#'*27 + 'Backward Elemination' + '#'*27)
    print(regressor_OLS.summary())
    print('#'*75)
    print("Here from OLS summary we can see none of the feature is having P value > 0.05 ")
    print("Hence all features seems to be significant, nothing can be dropped. ")
    print('#'*75)
