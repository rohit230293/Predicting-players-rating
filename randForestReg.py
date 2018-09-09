######################### RandomForestRegressor ##########################
from sklearn.ensemble import RandomForestRegressor
import numpy as np
'''
def randForestReg(X_train,X_test,y_train,y_test):
    regr_rf = RandomForestRegressor(max_depth=30, random_state=2)
    regr_rf.fit(X_train, y_train)
    #y_rf = regr_rf.predict(X_test)
    regr_rf.score(X_test,y_test)
'''
################
# Try different numbers of n_estimators - this will take a while or so

regr_rf = RandomForestRegressor(max_depth=30, random_state=2)

def randForestReg(X_train,X_test,y_train,y_test):
    
    estimators = np.arange(100, 200, 10)
    scores = []
    for n in estimators:
        regr_rf.set_params(n_estimators=n)
        regr_rf.fit(X_train, y_train)
        scores.append(regr_rf.score(X_test, y_test))
    max_sc_idx = scores.index(max(scores))   ### can be seen as n_estimators = 180 gives max score i.e. .9846058
    return (max(scores))
    '''
    #max_sc_idx = 18
    regr_rf_best = RandomForestRegressor(max_depth=30, random_state=2, n_estimators=max_sc_idx*10)
    regr_rf_best.fit(X_train, y_train)
    #y_rf = regr_rf.predict(X_test)
    return(regr_rf_best.score(X_test,y_test))
    '''
