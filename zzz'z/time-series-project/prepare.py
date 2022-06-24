'''
this module takes in the superstore dataframe and returns tidy data. there are multiple options for wrangling your data.
'''

from imports import *

def prep_data(df):
    '''
    takes in the dataframe of data obtained from teh superstore database.
    cleans and prepares the dataframe by doing the following:
    - renames columns for convenience
    - casts date columns as datetime types
    - resets the dataframe index as a datetime (order_date)
    - adds engineered features:
        - profit-per-product
        - sales-per-product
    returns the cleaned dataframe.
    '''
    
    # rename columns: lower case, remove space
    for col in df.columns:
        df = df.rename(columns={col: col.lower().replace(' ', '_')})
        
    # cast date columns as datetime type
    df.order_date = pd.to_datetime(df.order_date)
    df.ship_date = pd.to_datetime(df.ship_date)
        
    # make datetime index
    df.index = df.order_date
    
    # add profit per product column
    df['profit_per_product'] = df.profit / df.quantity
    
    # add sales per product
    df['sales_per_product'] = df.sales / df.quantity
    
    return df


def remove_outliers(train, test, k, col_list):
    ''' 
    This function takes in a dataset split into two sample dataframes: train and test.
    It calculates an outlier range based on a given value for k, using the interquartile range 
    from the train sample. It then applies that outlier range to each of the samples, removing
    outliers from a given list of feature columns. The train, and test dataframes 
    are returned, in that order. 
    '''
    # Create a column that will label our rows as containing an outlier value or not
    train['outlier'] = False
    test['outlier'] = False
    for col in col_list:

        q1, q3 = train[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        train['outlier'] = np.where(((train[col] < lower_bound) | (train[col] > upper_bound)) & (train.outlier == False), True, train.outlier)
        test['outlier'] = np.where(((test[col] < lower_bound) | (test[col] > upper_bound)) & (test.outlier == False), True, test.outlier)

    # remove observations with the outlier label in each of the three samples
    train = train[train.outlier == False]
    train = train.drop(columns=['outlier'])
    
    test = test[test.outlier == False]
    test = test.drop(columns=['outlier'])

    # print the remaining 
    print(f'train\t n = {train.shape[0]}')
    print(f'test\t n = {test.shape[0]}')

    return train, test

def split_data(df):
    '''
    Splits data into train and test based on year.
    '''
    # set train df to acquire rows from 2014 to 2016
    train = df['2014':'2016']
    # set test to last year of dataset
    test = df['2017']
    # output the number of observations in each dataset
    print('train n=', len(train))
    print('test n=', len(test))
    return train, test
