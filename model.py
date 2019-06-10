# ttests, Kmeans, linear regression

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.stats as stats
from scipy.stats import pearsonr
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error


def get_numeric_columns(df, skipcolumns=[]):
    #   ''' arguments - (dataframe, optional list of strings)
    #   Purpose is to return a list of numeric columns from a dataframe
    #   second argument is optional - is a list of numeric columns you dont want included in the returned list '''df.fillna(value=0, inplace=True)   
    column_list = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_column_list = [x for x in column_list if x not in skipcolumns]
    return(numeric_column_list)

def filter_columns(df,listofcolumns):
    #   ''' arguments - (dataframe), columns to include in returned dataframe
    #  ''' 
    newdf = df.copy()
    col_list = df.columns 
    for column in col_list:
        if column not in listofcolumns:
            newdf.drop([column], axis=1, inplace=True)
    return newdf

def compare_ttests(df,dependentvar,skipcolumns=[]):
    # '''  arguments  - (dataframe, dependent variable column (string), columns to skip (list) is optional) 
    # Purpose is to run Ttests on the dependent variable value of all columns, separated by the mean of each column
    # Only results where pvalue is less than 0.05 are returned (others are dropped)
    # returns a dataframe sorted by the coefficient of the ttest for each variable  '''   
    tcolumns = get_numeric_columns(df, skipcolumns=[])
    results = []
    features = []
    pvalues = []
    for thiscolumn in tcolumns:
        if thiscolumn != dependentvar:
            if thiscolumn != skipcolumns:
                df1 = df[df[thiscolumn] < df[thiscolumn].mean()]
                df2 = df[df[thiscolumn] > df[thiscolumn].mean()]        
                this_tstat, this_pvalue = stats.ttest_ind(df1[dependentvar].dropna(),
                   df2[dependentvar].dropna())
                if this_pvalue < .05:
                    results.append(this_tstat)
                    features.append(thiscolumn)
                    pvalues.append(this_pvalue)    
    list_of_tuples = list(zip(features,results,pvalues))
    results_df = pd.DataFrame(list_of_tuples, columns = ['Variables','T-Stats','Pvalues'])
    results_df.sort_values('T-Stats', inplace=True)
    return results_df  

def create_kclusters(df,col_list,n_clusters,nameof_clustercolumn):
    # '''  pass a datframe, list of columns to cluster, and a target number of clusters
    #      returns the modified datagrame, the inertia of the cluster fit, and a color constant applied to each point
    #   first, remove 'cluster_target' in case calling this function iteratively
    #   function will return a df with the 'cluster_target' column, evaluated and appended to it 
    #  '''

    if nameof_clustercolumn in df.columns:     
        df.drop([nameof_clustercolumn],axis=1,inplace=True)
    cluster_df = df.copy()
    df_columns = df.columns

    for thiscolumn in df_columns:
        if thiscolumn not in col_list:
            cluster_df.drop([thiscolumn], axis=1, inplace=True)

    kmeans = KMeans(n_clusters, random_state=123)            
    kmeans.fit(cluster_df)
      
    cluster_df[nameof_clustercolumn] = kmeans.predict(cluster_df)        
    cluster_df.drop(col_list,axis=1,inplace=True)
    return_df = pd.concat([df,cluster_df], axis=1, join_axes=[df.index]) 
    return return_df

def lregression_test(df,xfeatures,yfeature,train_size):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score

    y = df[yfeature]
    X = df[xfeatures]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=123)
    
    from sklearn.preprocessing import quantile_transform

    X_train = pd.DataFrame(quantile_transform(X_train, n_quantiles=10, random_state=0, copy=True))
    X_test = pd.DataFrame(quantile_transform(X_test, n_quantiles=10, random_state=0, copy=True))

    ### X_train = pd.DataFrame(preprocessing.scale(X_train))
    ### X_test = pd.DataFrame(preprocessing.scale(X_test))

    ### fit the transformed features to Linear Regression
    lm1 = LinearRegression(fit_intercept=False) 
    lm1.fit(X_train, y_train)
    LinearRegression(copy_X=True, fit_intercept=False, n_jobs=None, normalize=False)
    
    ### predicting on test data-set
    y_test_predict = lm1.predict(X_test)
    
    ### Cross Validation
    cross_val_score = cross_val_score(lm1, X_train, y_train, cv=3)

    y_train_predicted = lm1.predict(X_train)
    
    ### evaluating the model on training dataset
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
    r2_train = r2_score(y_train, y_train_predicted)

    ### evaluating the model on test dataset
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
    r2_test = r2_score(y_test, y_test_predict)

    print("The model performance for the training set")
    print("-------------------------------------------")
    print("RMSE of training set is {}".format(rmse_train))
    print("R2 score of training set is {}".format(r2_train))

    print("\n")
    
    print("The cross validation for the training set")
    print("-------------------------------------------")
    print("Cross Validation of training set is {}".format(cross_val_score))
    
    print("\n")

    print("The model performance for the test set")
    print("-------------------------------------------")
    print("RMSE of test set is {}".format(rmse_test))
    print("R2 score of test set is {}".format(r2_test))
    
    print("\n")
    
    pd.DataFrame({'actual': y_train.recovery,
                  'pm1': y_train_predicted.ravel()})\
                  .melt(id_vars=['actual'], var_name='model', value_name='prediction')\
                  .pipe((sns.relplot, 'data'), x='actual', y='prediction')
    
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], c='black', ls=':')
    plt.xlabel('Actual Recovery')
    plt.ylabel('Predicted Recovery')
    plt.title('Linear Regression: Predicted vs. Actual Recovery Amounts')
    
    return


def rregression_test(df,xfeatures,yfeature,train_size):
    from sklearn.linear_model import Ridge
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score
    
    '''
    Creates a ridge regression model for the given data frame, x features, y feature, and train size
    '''

    y = df[yfeature]
    X = df[xfeatures]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=123)

    from sklearn.preprocessing import quantile_transform

    X_train = pd.DataFrame(quantile_transform(X_train, n_quantiles=10, random_state=0, copy=True))
    X_test = pd.DataFrame(quantile_transform(X_test, n_quantiles=10, random_state=0, copy=True))
    
    ### X_train_scaled = pd.DataFrame(preprocessing.scale(X_train))
    ### X_test_scaled = pd.DataFrame(preprocessing.scale(X_test))

    ### fit the transformed features to Ridge Regression
    reg = linear_model.Ridge(alpha=.5)
    reg.fit(X_train, y_train)
    Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=123, solver='auto', tol=0.001)
    
    ### predicting on test data-set
    y_test_predict = reg.predict(X_test)
    
    ### Cross Validation
    cross_val_score = cross_val_score(reg, X_train, y_train, cv=3)

    y_train_predicted = reg.predict(X_train)
    
    ### evaluating the model on training dataset
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
    r2_train = r2_score(y_train, y_train_predicted)

    ### evaluating the model on test dataset
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
    r2_test = r2_score(y_test, y_test_predict)

    print("The model performance for the training set")
    print("-------------------------------------------")
    print("RMSE of training set is {}".format(rmse_train))
    print("R2 score of training set is {}".format(r2_train))

    print("\n")
    
    print("The cross validation for the training set")
    print("-------------------------------------------")
    print("Cross Validation of training set is {}".format(cross_val_score))
    
    print("\n")

    print("The model performance for the test set")
    print("-------------------------------------------")
    print("RMSE of test set is {}".format(rmse_test))
    print("R2 score of test set is {}".format(r2_test))
    
    print("\n")
    
    pd.DataFrame({'actual': y_train.recovery,
                  'pm1': y_train_predicted.ravel()})\
                  .melt(id_vars=['actual'], var_name='model', value_name='prediction')\
                  .pipe((sns.relplot, 'data'), x='actual', y='prediction')
    
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], c='black', ls=':')
    plt.xlabel('Actual Recovery')
    plt.ylabel('Predicted Recovery')
    plt.title('Ridge Regression: Predicted vs. Actual Recovery Amounts')

    return


def polynomial_regression_model(df, xfeatures, yfeature, train_size, file_name):
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score
    
    '''
    Creates a polynomial regression model for the given degree of 2
    '''

    y = df[yfeature]
    X = df[xfeatures]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=123)

    from sklearn.preprocessing import quantile_transform

    X_train = pd.DataFrame(quantile_transform(X_train, n_quantiles=10, random_state=0, copy=True))
    X_test = pd.DataFrame(quantile_transform(X_test, n_quantiles=10, random_state=0, copy=True))
    
    ### X_train = pd.DataFrame(preprocessing.scale(X_train))
    ### X_test = pd.DataFrame(preprocessing.scale(X_test))
  
    poly_features = PolynomialFeatures(degree=2)

    ### transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)

    ### fit the transformed features to Linear Regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    
    ### predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)
    
    ### Cross Validation
    cross_val_score = cross_val_score(poly_model, X_train, y_train, cv=3)

    ### predicting on test data-set
    y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))

    ### evaluating the model on training dataset
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
    r2_train = r2_score(y_train, y_train_predicted)

    ### evaluating the model on test dataset
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
    r2_test = r2_score(y_test, y_test_predict)

    print("The model performance for the training set")
    print("-------------------------------------------")
    print("RMSE of training set is {}".format(rmse_train))
    print("R2 score of training set is {}".format(r2_train))

    print("\n")
    
    print("The cross validation for the training set")
    print("-------------------------------------------")
    print("Cross Validation of training set is {}".format(cross_val_score))
    
    print("\n")

    print("The model performance for the test set")
    print("-------------------------------------------")
    print("RMSE of test set is {}".format(rmse_test))
    print("R2 score of test set is {}".format(r2_test))
    
    print("\n")
    
    pd.DataFrame({'actual': y_train.recovery,
                  'pm1': y_train_predicted.ravel()})\
                  .melt(id_vars=['actual'], var_name='model', value_name='prediction')\
                  .pipe((sns.relplot, 'data'), x='actual', y='prediction')
    
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], c='black', ls=':')
    plt.xlabel('Actual Recovery')
    plt.ylabel('Predicted Recovery')
    plt.title('Polynomial Regression: Predicted vs. Actual Recovery Amounts')

    pred_df = pd.DataFrame(y_test.recovery)
    pred_df = pred_df.reset_index(drop=True)
    test_pred_df = pd.DataFrame(y_test_predict)
    df_preds = pd.merge(pred_df, test_pred_df, how='left', left_index=True, right_index=True)
    df_preds.columns = ['actual', 'predicted']

    return df_preds.to_excel(file_name +'.xlsx')

    return

def lasso_regression_test(df,xfeatures,yfeature,train_size):
    from sklearn.linear_model import LinearRegression
    from sklearn import linear_model
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score

    y = df[yfeature]
    X = df[xfeatures]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=123)

    from sklearn.preprocessing import quantile_transform

    X_train = pd.DataFrame(quantile_transform(X_train, n_quantiles=10, random_state=0, copy=True))
    X_test = pd.DataFrame(quantile_transform(X_test, n_quantiles=10, random_state=0, copy=True))

    ### X_train = pd.DataFrame(preprocessing.scale(X_train))
    ### X_test = pd.DataFrame(preprocessing.scale(X_test))

    clf = linear_model.Lasso(alpha=0.1) 
    clf.fit(X_train, y_train)
    linear_model.Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000, 
          normalize=False, positive=False, precompute=False, random_state=None, 
          selection='cyclic', tol=0.0001, warm_start=False)
    
    ### predicting on test data-set
    y_test_predict = clf.predict(X_test)
    
    ### Cross Validation
    cross_val_score = cross_val_score(clf, X_train, y_train, cv=3)

    y_train_predicted = clf.predict(X_train)
    
    ### evaluating the model on training dataset
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
    r2_train = r2_score(y_train, y_train_predicted)

    ### evaluating the model on test dataset
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
    r2_test = r2_score(y_test, y_test_predict)

    print("The model performance for the training set")
    print("-------------------------------------------")
    print("RMSE of training set is {}".format(rmse_train))
    print("R2 score of training set is {}".format(r2_train))

    print("\n")
    
    print("The cross validation for the training set")
    print("-------------------------------------------")
    print("Cross Validation of training set is {}".format(cross_val_score))
    
    print("\n")

    print("The model performance for the test set")
    print("-------------------------------------------")
    print("RMSE of test set is {}".format(rmse_test))
    print("R2 score of test set is {}".format(r2_test))
    
    print("\n")
    
    pd.DataFrame({'actual': y_train.recovery,
                  'pm1': y_train_predicted.ravel()})\
                  .melt(id_vars=['actual'], var_name='model', value_name='prediction')\
                  .pipe((sns.relplot, 'data'), x='actual', y='prediction')
    
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], c='black', ls=':')
    plt.xlabel('Actual Recovery')
    plt.ylabel('Predicted Recovery')
    plt.title('Lasso Regression: Predicted vs. Actual Recovery Amounts')
    
    return

def run_models(df,xfeatures,yfeature,train_size, file_name):
    '''
    Funtion to run all models
    '''
    print('Linear Regression Model:')
    lregression_test(df, xfeatures, yfeature, train_size)
    print('\n')
    print('Ridge Regression Model:')
    rregression_test(df, xfeatures, yfeature, train_size)
    print('\n')
    print('Polynomial Regression Model:')
    polynomial_regression_model(df, xfeatures, yfeature, train_size, file_name)
    
    return