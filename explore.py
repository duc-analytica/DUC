import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def encode_scale(df):   
# add any columns to be ignore, this function will ignore any column in this list PLUS any column that contains a substring in this list    
    columns_to_ignore = ['multi_well_lease','direction','status','major_phase','prod_method','frac_fluid_type','api14','encoded_','scaled_','recovery','vintage_bin','sur_lat','sur_long','peak_boepd','oil_hist','gas_hist','ip90_boeqpd','well_id','months_active','_prod','lease_name','well_number']
    numeric_columns = df.select_dtypes(exclude=['object']).columns.tolist()
    object_columns = df.select_dtypes(include=['object']).columns.tolist()
    columns_to_scale = []
    columns_to_encode = []
# numeric columns will only have to be scaled    
    for scaler in numeric_columns:
        foundignore = False
        for excludecol in columns_to_ignore:
            if excludecol in scaler:
                foundignore = True
                break
        if foundignore == False:
            columns_to_scale.append(scaler)
# object columns have to be encoded as well as scaled             
    for scaler in object_columns:
        foundignore = False
        for excludecol in columns_to_ignore:
            if excludecol in scaler:
                foundignore = True
                break
        if foundignore == False:
# add to scale and encode lists            
            columns_to_encode.append(scaler)
            columns_to_scale.append(scaler)
# assumes that all nulls have already been removed or filled at this point
    encoder = LabelEncoder()    
    for col in columns_to_encode:       
        encoder.fit(df[col])
        new_encoded_column = 'encoded_'+col
        df[new_encoded_column] = encoder.transform(df[col])    
    for col in columns_to_scale:
        new_scaled_column = 'scaled_'+col
        if col in columns_to_encode:
            new_encoded_column = 'encoded_'+col
            df[new_scaled_column] = (df[new_encoded_column] - df[new_encoded_column].min()) / (df[new_encoded_column].max() - df[new_encoded_column].min())
        else:            
            df[new_scaled_column] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return(df)