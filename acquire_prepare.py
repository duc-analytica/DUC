import pandas as pd
import numpy as np

### Select Data ###

def acquire_oil():
    '''
    Selected Data from Excel File
    '''
    return pd.read_excel('CapstoneData.xlsx').infer_objects()

### Process Data ###

def clean_columns(df):
    '''
    Function that takes the initial dataframe columns and cleans them by lowercasing, space replacing, and placing '_' in place of space
    '''
    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.replace(' ', '_')
    df.rename(index=str, columns={'majorphase':'major_phase'}, inplace=True)
    
    return df

def reclass_status(df):
    '''   consolidate all the different 'Status' labels to either 'Active' 
       when 'last_prod' date is after September 2018
      - OR -     'Inactive'  when 'last_prod' date is equal to or prior to September 2018
     '''
    df['status'] = np.where((df.last_prod > '2018-9-1'), 'Active', 'Inactive')
    return df

def fill_inactive_eurs(df):
    '''   look for null EURs in 'Inactive' wells and populate those null values with 
       'hist' values   (cumulative oil or gas volumes)
    '''
    df['oil_eur'] = np.where(  ( (df.status == 'Inactive') & (df.oil_eur.isna()) ) , df.oil_hist, df.oil_eur  )
    df['gas_eur'] = np.where(  ( (df.status == 'Inactive') & (df.gas_eur.isna()) ) , df.gas_hist, df.gas_eur  )
    return df

def drop_na(df):
    '''
    Function that drops null values in specific columns
    '''
    df.dropna(subset=['oil_eur', 'gas_eur'], inplace=True)
    
    return df

def numeric_to_category(df):
    '''
    Function to change selected columns into a category type
    '''
    cols = ['type', 'status', 'major_phase', 'prod_method', 'frac_fluid_type']
    
    df[cols] = df[cols].astype('category')
    
    return df

### Transform Data ###

def select_rows(df):
    '''
    Funtion to take out unwanted rows
    '''
    df = df[df.status != 'Injection']
    df = df[df.major_phase != 'INJ']
    df = df[df.major_phase != 'SWD']
    df = df[df.lateral_len != 0]
    df = df[df.lateral_len < 14000]
    
    return df

def post_year(df):
    '''
    Funtion that removes all data before year 1940
    '''
    df = df[df['first_prod'].dt.year > 1940]
    df = df[df['last_prod'].dt.year > 1940]
    
    return df

def feature_engineer(df):
    '''
    Function that creates four new features:
        recovery: total number of barrels of either natural gas/oil
        recovery_per_foot: amount of recovery per foot using the pipes lateral length
        months_active: total number of months active
        recover_per_month: amount each API gained per month
    '''
    #  recovery units are     mboeq         (thousand barrel oil equivalents)   
    df['recovery'] = df.oil_eur + df.gas_eur/6
    #  recovery_per_foot units are  boe/ft       (barrels (boe) per foot)    
    df['recovery_per_foot'] = df['recovery']/df['lateral_len'] * 1000
    
    df['months_active'] = (df.last_prod.dt.to_period('M') - df.first_prod.dt.to_period('M')).astype(int)
    df['months_active'].replace('0', '1', inplace=True)

    # recovery_per_month  units  are     bbls/month    (barrels per month)
    df['recovery_per_month'] = (df.recovery * 1000) / df.months_active
    
    return df

def remove_columns(df):
    '''
    Function that removes columns we no longer need:  ,   gg - got rid of entity_reserve_column
    '''
    cols_to_remove = [ 'oil_eur', 'gas_eur']
    
    df = df.drop(columns=cols_to_remove)
    
    return df

def prep_data(df):
    '''
    Function that combines all functions and produces a final dataframe in a csv
    '''
    df = clean_columns(df)
    df = post_year(df)
    df = reclass_status(df)
    df = fill_inactive_eurs(df)
    df = drop_na(df)
    df = numeric_to_category(df)
    df = select_rows(df)
    df = feature_engineer(df)
    df = remove_columns(df)

    df.to_csv('cleaned_oil_df.csv', index=False)
    
    return df