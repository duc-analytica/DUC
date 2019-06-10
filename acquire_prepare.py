import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

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
    df = df[df.gor_hist < 20000]
    df = df[df.frac_fluid_gpf < 3000]
    df = df[df.proppant_ppf < 4000]
    
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
    df = df[df.recovery < 700]
    #  recovery_per_foot units are  boe/ft       (barrels (boe) per foot)    
    df['recovery_per_foot'] = df['recovery']/df['lateral_len'] * 1000
    df = df[df.recovery_per_foot < 1000]
    
    df['months_active'] = (df.last_prod.dt.to_period('M') - df.first_prod.dt.to_period('M')).astype(int)
    df['months_active'].replace('0', '1', inplace=True)

    # recovery_per_month  units  are     bbls/month    (barrels per month)
    df['recovery_per_month'] = (df.recovery * 1000) / df.months_active

    # lateral_class are binned lengths per 1000 feet
    labels=['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen']

    df['lateral_class'] = pd.cut(df.lateral_len, [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000,  8000, 9000, 10000, 11000, 12000, 13000, 14000], include_lowest=True)
    
    return df

def remove_columns(df):
    '''
    Function that removes columns we no longer need:  ,   gg - got rid of entity_reserve_column
    '''
    cols_to_remove = [ 'oil_eur', 'gas_eur', 'oil_gravity']
    
    df = df.drop(columns=cols_to_remove)
    
    return df

def vintage_clean_types(df):
    '''
    Function that creates a column that houses year of first production, then changes all pre 1996 pumps
    to vertical based on the fact that horizontal drilling started in 1996.  The final step is to change
    types that are 'other' to vertical or horizontal based on lateral length.
    '''
    df['vintage'] = df.first_prod.dt.year
    df.type.loc[df['vintage'] < 1990] = 'Vertical'
    df.type.loc[(df['lateral_len'] < 1600) & (df.type == 'Other')] = 'Vertical'
    df.type.loc[(df['lateral_len'] > 1600) & (df.type == 'Other')] = 'Horizontal'

    return df

def bin_vintage(df):
    '''
    Function to bin vintage into 5 groups
    '''
    # vintage binned by era's 
    year_labels=[1, 2, 3, 4, 5]

    df['vintage_bin'] = pd.cut(df.vintage, [0, 1970, 1990, 2000, 2013, 2020], include_lowest=True, labels=year_labels)
    df['vintage_bin'] = df['vintage_bin'].astype(str).astype(int)

    return df

def remove_integrity_issues_lat_length_type(df):
    '''
    Using domain knowledge, we have decided to remove wells that are over 1,600 lateral length and classified
    as vertical as well as remove wells under 600 lateral length and classified as horizontal as these appear
    to be data integrity issues.
    '''
    df = df[~((df.lateral_len > 1600) & (df.type == 'Vertical'))]
    df = df[~((df.lateral_len < 600) & (df.type == 'Horizontal'))]

    return df

def remove_non_permian_counties(df):
    '''
    Function to remove counties that are not in geographic proximity of the Permian Basin.
    '''
    df = df[~(df.county.isin(['ATASCOSA', 'BEE', 'DE WITT', 'DIMMIT', 'FRIO', 'GOLIAD', 'KARNES','LIVE OAK', 'MC MULLEN', 'WILSON']))]

    return df

def minimize_operators(df):
    '''
    Function to change operators for wells not in the top 9 to other in an effort to reduce the number of operators.
    '''
#     top_10 = df.oper.value_counts()[:9].index.tolist()
#     df.oper[~df.oper.isin(top_10)] = 'OTHER'
    '''
    Function to change operators for wells to root of the operator name.
    '''
    df.oper = df.oper.str.replace(r'(^.*APACHE.*$)','apache')
    df.oper = df.oper.str.replace(r'(^.*PIONEER.*$)','pioneer')
    df.oper = df.oper.str.replace(r'(^.*CALLON.*$)','callon')
    df.oper = df.oper.str.replace(r'(^.*CHEVRON.*$)','chevron')
    df.oper = df.oper.str.replace(r'(^.*COG.*$)','cog')
    df.oper = df.oper.str.replace(r'(^.*CONCHO.*$)','concho')
    df.oper = df.oper.str.replace(r'(^.*ENDEAVOR.*$)','endeavor')
    df.oper = df.oper.str.replace(r'(^.*GLOSSOP.*$)','glossop')
    df.oper = df.oper.str.replace(r'(^.*OXY.*$)','oxy')
    df.oper = df.oper.str.replace(r'(^.*OCCIDENTAL.*$)','oxy')
    df.oper = df.oper.str.replace(r'(^.*PARSLEY.*$)', 'parsley')
    df.oper = df.oper.str.replace(r'(^.*parsley.*$)', 'parker & parsley')
    df['oper'] = df['oper'].map(lambda oper: oper.lower())

    return df

def fill_zero(df):
    '''
    Function to replace all null values with zero or unknown
    '''
    df['gor_hist'].fillna(value=0, inplace=True)
    df['frac_fluid_type'] = df['frac_fluid_type'].replace(np.nan, 'Unknown')
    df.frac_fluid_type.loc[(df['frac_fluid_type'] == 'None')] = 'Unknown'
    df['type'] = df['type'].replace(np.nan, 'Unknown')
    df['status'] = df['status'].replace(np.nan, 'Unknown')
    df['major_phase'] = df['major_phase'].replace(np.nan, 'Unknown')
    df['prod_method'] = df['prod_method'].replace(np.nan, 'Unknown')
    df['lateral_class'] = df['lateral_class'].replace(np.nan, 'Unknown')
    # change landing_depth NaNs to median of that formation
    med_Spr = df.landing_depth[df.formation == 'SPRABERRY'].median()
    df.landing_depth.fillna(value=med_Spr, inplace=True)
    # change frac_stages NaNs of vertically-drilled wells to 1
    middle = df['frac_stages'][(df.type == 'Horizontal')].median()
    condition1 = (df.frac_stages.isnull()) & (df.type == 'Vertical')
    df['frac_stages'] = np.where((condition1), 1, df['frac_stages'])
    # change frac_stages NaNs of horizontally-drilled wells to median of the horizontally-drilled wells
    condition2 = (df.frac_stages.isnull()) & (df.type == 'Horizontal')
    df['frac_stages'] = np.where((condition2), middle, df['frac_stages'])
    # change oil_gravity NaNs to 0.
    df['oil_gravity'] = np.where(df.oil_gravity.isnull(), 0, df['oil_gravity'])
    
    return df

def rename_cols(df):
    '''
    Rename columns to more industry specific names
    '''    
    df.rename(index=str, columns={'lateral_len':'gross_perfs'}, inplace=True)
    df.rename(index=str, columns={'landing_depth':'tvd'}, inplace=True)
    df.rename(index=str, columns={'type':'direction'}, inplace=True)

def encode_scale(df):    
# add any columns to be ignore, this function will ignore any column in this list PLUS any column that contains a substring in this list    
    columns_to_ignore = ['recovery','clusterid','recovery_per_foot','recovery_per_month','multi_well_lease','status','major_phase','prod_method','api14','encoded_','scaled_','recovery','vintage_bin','sur_lat','sur_long','peak_boepd','oil_hist','gas_hist','ip90_boeqpd','well_id','months_active','_prod','lease_name','well_number','sub_basin']
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    object_columns = df.select_dtypes(include='object').columns.tolist()
    category_columns = df.select_dtypes(include='category').columns.tolist()
    columns_to_encode = category_columns + object_columns
    i = 0
    for i in range(len(numeric_columns) - 1, -1, -1):
        element = numeric_columns[i]
        if element in columns_to_ignore:
            del numeric_columns[i]
        elif 'encoded_' in element:
            del numeric_columns[i]
        elif 'scaled_' in element:
            del numeric_columns[i]
    
    for i in range(len(columns_to_encode) - 1, -1, -1):
        element = columns_to_encode[i]
        if element in columns_to_ignore:
            del columns_to_encode[i]
        elif 'encoded_' in element:
            del columns_to_encode[i]
        elif 'scaled_' in element:
            del columns_to_encode[i]            
# numeric columns will only have to be scaled 
# object columns have to be encoded and scaled 
# assumes that all nulls have already been removed or filled at this point
    for col in columns_to_encode: 
        encoder = LabelEncoder() 
        encoder.fit(df[col])
        new_encoded_column = 'encoded_'+col
        df[new_encoded_column] = encoder.transform(df[col])   
    for col in columns_to_encode:
        new_scaled_column = 'scaled_'+col
        new_encoded_column = 'encoded_'+col
#         df[new_scaled_column] = pd.DataFrame(preprocessing.scale(df[new_encoded_column]))
        df[new_scaled_column] = (df[new_encoded_column] - df[new_encoded_column].min()) / (df[new_encoded_column].max() - df[new_encoded_column].min())
#         print('scaled value in for loop for columns_to_encode:')
#         print(df[new_scaled_column])

# tried removing these API14 and Well Id from from scaling to prevent warning about int64 being converted to float64, but still getting warning
    print()
    for col in numeric_columns: 
        new_scaled_column = 'scaled_'+col
#         df[new_scaled_column] = pd.DataFrame(preprocessing.scale(df[col]))
        df[new_scaled_column] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
#         print('scaled value in for loop for columns_to_encode:')
#         print(df[new_scaled_column])

    return(df)

def prep_data(df):
    #''' Function that combines all functions and produces a final dataframe in a csv
    #    basin options = all,central platform, delaware, midland
    #    direc options = 'all', 'horizontal', 'vertical'
    #    tvdmin  (numeric)  - selects observations greater than tvdmin  (0 means no minimum)
    #    tvdmax  (numeric)  - selects observations less than tvdmin  (0 means no maximum)
    #    gperfmin  (numeric)  - selects observations greater than gperfmin  (0 means no minimum)
    #    gperfmax  (numeric)  - selects observations less than gperfmax  (0 means no maximum)    

    # (**Automation Features**),basin='all',direc='all',tvdmin=0,tvdmax=0,gperfmin=0,gperfmax=0,multiwell='all',clusterid=99
    #    multi-well options - 'all', 'true', 'false'
    #    clusterid options - 99=all, or input specific numeric id value
    #    here's an example
    #    prep_data(df,'midland','horizontal',0,7000,4800,0,'false',99)
    #    returns a dataframe for midland sub_basin, horizontal wells only, tvd less than 7000, 
    #         gross_perfs greater than 4800, single well leases only, any cluster  
    #         resulting dataset is less than 100 rows
    #'''
    df = clean_columns(df)
    df = post_year(df)
    df = reclass_status(df)
    df = fill_inactive_eurs(df)
    df = drop_na(df)
    df = select_rows(df)
    df = feature_engineer(df)
    df = remove_columns(df)
    df = vintage_clean_types(df)
    df = bin_vintage(df)
    df = remove_integrity_issues_lat_length_type(df)
    df = remove_non_permian_counties(df)
    df = minimize_operators(df)
    df = fill_zero(df)
    df = numeric_to_category(df)
    rename_cols(df)
    # filter out subsets of dataframe based on function parameters, this needs to happen
    #  before scaling the data
    '''
    if basin !='all':
        df = df[(df.sub_basin.str.lower() == basin.lower())]
    if multiwell.lower() !='all':
        if multiwell.lower() == 'true':   
            df = df[(df.multi_well_lease == True)]    
        if multiwell.lower() == 'false':   
            df = df[(df.multi_well_lease == False)]        
    if direc !='all':
        df = df[(df.direction.str.lower() == direc.lower())] 
    if tvdmin > 0:
        df = df[(df.tvd > tvdmin)] 
    if tvdmax > 0:
        df = df[(df.tvd < tvdmax)] 
    if gperfmin > 0:
        df = df[(df.gross_perfs > gperfmin)]   
    if gperfmax > 0:
        df = df[(df.gross_perfs < gperfmax)]   
    if clusterid != 99:
        df = df[(df.clusterid == clusterid)] 
    if len(df) < 100:
        print(' WARNING- Number of Observations is too low: ',len(df))
        '''

    df = encode_scale(df)   
    df.to_csv('cleaned_oil_df.csv', index=False)
    
    return df