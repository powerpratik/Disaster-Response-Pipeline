#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline Preparation

# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data():
    # load messages dataset
    messages = pd.read_csv('messages.csv')
    messages.head(20)

    # load categories dataset
    categories = pd.read_csv('categories.csv')
    categories.head(20)

    # merge datasets
    df = pd.merge(left=messages,right=categories,on='id')
    df.head(20)

    # ### . Split `categories` into separate 36 category columns.
    
    categories =categories['categories'].str.split(';',expand=True)
    categories.head()

    # select the first row of the categories dataframe
    row = categories[0:1]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda val:val[0][0:-2],axis=0).tolist()
    print(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()

    # ###  Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    categories.head()


    # drop the original categories column from `df` because it's no longer needed
    df.drop(labels=['categories'],axis=1,inplace=True)
    df.head()
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    df.head()
    return df


def clean_data(df):
    # ### 6. Remove duplicates along with handling null and infinity values

    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.fillna(0,inplace=True)
    df.drop_duplicates()

def save_data():
# ### 7. Save the clean dataset into an sqlite database.
engine = create_engine('sqlite:///InsertDatabaseName.db')
df.to_sql('InsertTableName', engine, index=False)




