#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline Preparation

# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_path,categories_path):
    # load messages dataset
    messages = pd.read_csv(messages_path)
    messages.head(20)

    # load categories dataset
    categories = pd.read_csv(categories_path)
    categories.head(20)

    # merge datasets
    df = pd.merge(left=messages,right=categories,on='id')
    
    # ### . Split `categories` into separate 36 category columns.
    
    categories =categories['categories'].str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories[0:1]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda val:val[0][0:-2],axis=0).tolist()
    print(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames

    # ###  Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df` because it's no longer needed
    df.drop(labels=['categories'],axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    print(df.head())
    return df


def clean_data(df):
    # ### 6. Remove duplicates along with handling null and infinity values

    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.fillna(0,inplace=True)
    df.drop_duplicates()

def save_data(df,database_path):
    # ### 7. Save the clean dataset into an sqlite database.
    engine = create_engine('sqlite:///'+database_path)
    df.to_sql('table', engine, index=False)

def main():
    if len(sys.argv)==4:
         messages_path, categories_path, database_path = sys.argv[1:]
         print('Loading Data:....\n MESSAGES:{}\n CATEGORIES:{}\n'.format(messages_path,categories_path))
         
         print('Data Frame with 5 rows ')
         df=load_data(messages_path,categories_path)
         
         print('Cleaning Data')
         df= clean_data(df)

         print('Rows after Cleaning data')
         print(df.head())

         print('Saving Data to....\n    DataBase:{}'.format(database_path))
         save_data(df,database_path)

         print('Successfully saved the data to DataBase..')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')
              
if __name__=='__main__':
    main()