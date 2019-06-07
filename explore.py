import pandas as pd
import numpy as np
import xgboost as xgb
from model import get_scaled_df

def xgb_rank(df,target_variable,feature_percent=80,mode='gain'):
    # ''' pass it the dataframe and the target variable,  
    #   returns a sorted feature list, and a sorted scaled feature list, and a dataframe
    #    the two returned lists only have the variables that satisfy the cumulative percentage limit
    #       the dataframe has all features
    #      in decending order as well as a dataframe showing all cumulative percent rankings
    #      feature_percent is the optional cut-off (default is 80 percent) for features
    #      mode is optional,  default is 'gain' (importance),  other values are
    #       'weight'
    # '''

    scaled_df = get_scaled_df(df) 
    scaled_df = scaled_df.drop('recovery',1) 
    xgb_params = {'max_depth': 8,'seed' : 123}
    dtrain = xgb.DMatrix(scaled_df, target_variable, feature_names=scaled_df.columns.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)
    importance_dict = model.get_score(importance_type=mode)
    sorted_importance_dict = sorted(importance_dict.items(), key=lambda kv: kv[1])
    importance_df = pd.DataFrame.from_dict(sorted_importance_dict)
    importance_df.columns = ['feature',mode] 
    importance_df.sort_values(mode, inplace = True) 
    importance_df['rank'] = importance_df[mode].rank(ascending = False)
    importance_df.sort_values('rank', inplace = True) 
    importance_df.set_index('rank', inplace = True)
    importance_df.reset_index(inplace=True) 
    importance_df[mode] = importance_df[mode].apply(lambda x: round(x, 2))
    importance_df['cum_sum'] = round(importance_df[mode].cumsum(),2)
    importance_df['cum_perc'] = round(100*importance_df.cum_sum/importance_df[mode].sum(),2)
    feature_list = []
    scaled_features = [] 
    for i in range((importance_df.shape[0])): 
        feature_name = importance_df.iloc[i,1].replace('scaled_','')
        scaled_name = 'scaled_' + feature_name
        importance_df.iloc[i,1] = feature_name
        cum_percent = importance_df.iloc[i,4]
        if cum_percent > feature_percent:
            break
        else:
            feature_list.append(feature_name)
            scaled_features.append(scaled_name)
    return feature_list, scaled_features, importance_df, 
