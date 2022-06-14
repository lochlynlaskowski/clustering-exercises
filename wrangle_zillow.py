import pandas as pd
import numpy as np
import os
from env import get_db_url
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

def get_zillow_data():
    '''Returns a dataframe of all single family residential properties from 2017. Initial 
    query is from the Codeup database. File saved as CSV and called upon after initial query.'''
    filename = "zillow.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        sql = '''SELECT *
        FROM properties_2017
        JOIN (SELECT parcelid, MAX(transactiondate) AS max_transactiondate
                FROM predictions_2017
                GROUP BY parcelid) pred USING(parcelid)
        JOIN predictions_2017 USING (parcelid)
        LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
        LEFT JOIN propertylandusetype USING (propertylandusetypeid)
        LEFT JOIN airconditioningtype USING (airconditioningtypeid)
        LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
        LEFT JOIN storytype USING (storytypeid)
        LEFT JOIN unique_properties USING (parcelid)
        LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
        WHERE propertylandusetype.propertylandusedesc = 'Single Family Residential'
        AND predictions_2017.transactiondate LIKE '2017%%'
        AND properties_2017.latitude IS NOT NULL
        AND properties_2017.longitude IS NOT NULL;'''

    df = pd.read_sql(sql, get_db_url('zillow'))
    return df


def find_missing_values(df):
    column_name = []
    num_rows_missing = []
    pct_rows_missing = []

    for column in df.columns:       
        num_rows_missing.append(df[column].isna().sum())
        pct_rows_missing.append(df[column].isna().sum()/ len(df))
        column_name.append(column)
    data = {'column_name':column_name, 'num_rows_missing': num_rows_missing, 'pct_rows_missing': pct_rows_missing}
    return pd.DataFrame(data, index=None)


def handle_missing_values(df, prop_required_column, prop_required_row):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


def handle_outliers(df, cols, k):
    # Create placeholder dictionary for each columns bounds
    bounds_dict = {}

    # get a list of all columns that are not object type
    non_object_cols = df.dtypes[df.dtypes != 'object'].index


    for col in non_object_cols:
        # get necessary iqr values
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr

        #store values in a dictionary referencable by the column name
        #and specific bound
        bounds_dict[col] = {}
        bounds_dict[col]['upper_bound'] = upper_bound
        bounds_dict[col]['lower_bound'] = lower_bound

    for col in non_object_cols:
        #retrieve bounds
        col_upper_bound = bounds_dict[col]['upper_bound']
        col_lower_bound = bounds_dict[col]['lower_bound']

        #remove rows with an outlier in that column
        df = df[(df[col] < col_upper_bound) & (df[col] > col_lower_bound)]
    
    return df


def split_zillow_data(df):
    ''' This function splits the cleaned dataframe into train, validate, and test 
    datasets.'''

    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123) 
                                   
    return train, validate, test
