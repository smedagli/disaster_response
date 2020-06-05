import pandas as pd
from sqlalchemy import create_engine

from disaster_response import paths

pd.options.display.max_columns =25
pd.options.display.width = 2500

categories = pd.read_csv(paths.categories_path)
messages = pd.read_csv(paths.messages_path)
df = pd.merge(messages, categories, on='id')

category_colnames = [a.split('-')[0] for a in categories.categories.str.split(';', expand=True).iloc[0]]
categories_clean = pd.DataFrame(columns=category_colnames)

for column in categories_clean:
    categories_clean[column] = df.categories.apply(lambda x: x.split(column, 1)[-1][1: 2] if column in x else 0)

df.drop('categories', axis=1, inplace=True)
# concatenate the original dataframe with the new `categories` dataframe
df = pd.merge(df, categories_clean, left_index=True, right_index=True, how='outer').fillna(0)
df.drop_duplicates(inplace=True)

engine = create_engine(f'sqlite:///{paths.sql_path}')
df.to_sql('table1', engine, index=False, if_exists='replace')