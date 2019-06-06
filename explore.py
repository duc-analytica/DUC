import pandas as pd
import numpy as np
import xgboost as xgb
from model import get_scaled_df

def xgb_rank(df,target_variable):
    # ''' pass it the dataframe and the target variable,  returns a sorted list 
    #      decending order 
    # '''
    scaled_df = get_scaled_df(df)  
    scaled_columns = scaled_df.columns.values.tolist()
    xgb_params = {'max_depth': 8,'seed' : 123}
    dtrain = xgb.DMatrix(scaled_df, target_variable, feature_names=scaled_df.columns.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)
    importance_dict = model.get_score(importance_type='gain')
    sorted_importance_dict = sorted(importance_dict.items(), key=lambda kv: kv[1])
    feature_list = [i[0] for i in sorted_importance_dict]
    feature_list.reverse()
    feature_list = [i.replace('scaled_','') for i in feature_list]
    return feature_list[:7]