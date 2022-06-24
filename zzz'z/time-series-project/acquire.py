'''
this module pulls the superstore dataframe using a mysql query from the codeup db over mysql+pymysql protocol. creds saved in env. check .gitignore for security. 
'''


from imports import *

def get_data():
    '''
    reads superstore data from codeup mysql database, returns joined 
    df of all available columns
    '''
    # establish a filename for the local csv
    filename = 'superstore.csv'
    # check to see if a local copy already exists. 
    if os.path.exists(filename):
        print('Reading from local CSV...')
        # if so, return the local csv
        return pd.read_csv(filename)
    # otherwise, pull the data from the database:
    # establish database url
    url = env.get_db_url('superstore_db')
    sql = '''
        SELECT ord.*,
                cat.Category,
                prod.`Product Name`,
                cust.`Customer Name`,
                reg.`Region Name`
            FROM orders ord
                LEFT JOIN categories cat USING(`Category ID`)
                LEFT JOIN products prod USING(`Product ID`)
                LEFT JOIN customers cust USING(`Customer ID`)
                LEFT JOIN regions reg USING(`Region ID`)
        '''
    print('No local file exists\nReading from SQL database...')
    # query the database and return the resulting table as a pandas dataframe
    df = pd.read_sql(sql, url)
    # save the dataframe to the local directory as a csv
    print('Saving to local CSV... ')
    df.to_csv(filename, index=False)
    # return the resulting dataframe
    return df