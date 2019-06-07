# ttests, Kmeans, linear regression

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import scipy.stats as stats
from scipy.stats import pearsonr
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error


def get_scaled_df(df):
    numerics = ['int64', 'float64', 'float']

    scaled_df = df.select_dtypes(include=numerics)
    scaled_df = scaled_df.drop(columns=['api14', 'recovery' ,'proppant_ppf', 'frac_fluid_gpf', 
            'gross_perfs', 'frac_stages', 'oil_gravity', 'peak_boepd', 'oil_hist', 
            'gas_hist', 'gor_hist', 'ip90_boeqpd', 'tvd', 'sur_lat', 'sur_long', 
            'well_id', 'mid_point_lat', 'mid_point_long', 'recovery_per_foot', 
            'months_active', 'recovery_per_month', 'vintage', 'vintage_bin', 
            'encoded_direction', 'encoded_frac_fluid_type', 'encoded_county', 
            'encoded_oper', 'encoded_formation', 'encoded_lateral_class',])
    return scaled_df

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

    kmeans = KMeans(n_clusters)            
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
    
    X_train_scaled = pd.DataFrame(preprocessing.scale(X_train))
    X_test_scaled = pd.DataFrame(preprocessing.scale(X_test))
    
    y_train.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    train = pd.concat([X_train_scaled, y_train], axis=1)
    test = pd.concat([X_test_scaled, y_test], axis=1)

    lm1 = LinearRegression(fit_intercept=False) 
    lm1.fit(X_train, y_train)
    LinearRegression(copy_X=True, fit_intercept=False, n_jobs=None, normalize=False)
    
    cross_val_score = cross_val_score(lm1, X_train, y_train, cv=3)
    
    lm1_y_intercept = lm1.intercept_
    lm1_coefficients = lm1.coef_
    y_pred_lm1 = lm1.predict(X_train)
    
    mse = mean_squared_error(y_train, y_pred_lm1)
    r2 = r2_score(y_train, y_pred_lm1)
    
    return mse, r2, lm1.coef_, cross_val_score


def rregression_test(df,xfeatures,yfeature,train_size):
    from sklearn.linear_model import Ridge
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score

    y = df[yfeature]
    X = df[xfeatures]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=123)
    
    X_train_scaled = pd.DataFrame(preprocessing.scale(X_train))
    X_test_scaled = pd.DataFrame(preprocessing.scale(X_test))
    
    y_train.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    train = pd.concat([X_train_scaled, y_train], axis=1)
    test = pd.concat([X_test_scaled, y_test], axis=1)

    reg = linear_model.Ridge(alpha=.5)
    reg.fit(X_train, y_train)
    Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=123, solver='auto', tol=0.001)
    
    cross_val_score = cross_val_score(reg, X_train, y_train, cv=3)
    
    reg_y_intercept = reg.intercept_ 
    reg_coefficients = reg.coef_
    y_pred_reg = reg.predict(X_train)
    
    mse = mean_squared_error(y_train, y_pred_reg)
    r2 = r2_score(y_train, y_pred_reg)
    
    return mse, r2, reg.coef_, cross_val_score


def pregression_test(df,xfeatures,yfeature,train_size):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score

    y = df[yfeature]
    X = df[xfeatures]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=123)
    
    X_train_scaled = pd.DataFrame(preprocessing.scale(X_train))
    X_test_scaled = pd.DataFrame(preprocessing.scale(X_test))
    
    y_train.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    train = pd.concat([X_train_scaled, y_train], axis=1)
    test = pd.concat([X_test_scaled, y_test], axis=1)

    poly_features = PolynomialFeatures(degree=2)
    X_train_scaled = poly_features.fit_transform(X_train_scaled)

    model = Pipeline([('poly', PolynomialFeatures(degree=2)), 
        ('linear', LinearRegression(fit_intercept=False))])
    
    
    model = model.fit(X_train_scaled, y_train)
    model.named_steps['linear'].coef_
    
    cross_val_score = cross_val_score(model, X_train, y_train, cv=3)
    
    y_pred_poly = model.predict(X_train_scaled)
    mse = mean_squared_error(y_train, y_pred_poly)
    r2 = r2_score(y_train, y_pred_poly)
    
    return mse, r2, model.named_steps['linear'].coef_, cross_val_score