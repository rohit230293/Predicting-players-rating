# Applying the XGBoost
import xgboost

def xgb(X_train,X_test,y_train,y_test):
    regressor_xg = xgboost.XGBRegressor()
    regressor_xg.fit(X_train,y_train)
    #regressor_xg.predict(X_test)
    return(regressor_xg.score(X_test,y_test))
