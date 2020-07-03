"""
13/06/2020
Preprocessing of the data. Run the script as
```python data/process_data.py <messages_csv_path> <categories_csv_path> <database_db_path>```

will import messages and categories from the 2 .csv files, merge them and save into the .db file

See Also:
    data/process_data_script.py
"""

import sys
import os
import pandas as pd
from sqlalchemy import create_engine

import paths


def main(messages_path=paths.messages_path,
         category_path=paths.categories_path,
         sql_path=paths.sql_path,
         write=1) -> pd.DataFrame:
    """
    Merges messages and categories in a sql table and save it
    Args:
        messages_path: path of the .csv file of the messages
        category_path: path of the .csv file of the categories / labels
        sql_path: path of the .db file to save to save
        write: if 1, writes the file
    Returns:

    """
    categories = pd.read_csv(category_path)
    messages = pd.read_csv(messages_path)
    df = pd.merge(messages, categories, on='id')

    category_colnames = [a.split('-')[0] for a in categories.categories.str.split(';', expand=True).iloc[0]]
    categories_clean = pd.DataFrame(columns=category_colnames)
    for column in categories_clean:
        categories_clean[column] = df.categories.apply(lambda x: x.split(column, 1)[-1][1: 2] if column in x else 0)
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df, categories_clean, left_index=True, right_index=True, how='outer').fillna(0)
    df.drop_duplicates(inplace=True)

    if write:
        print(f"Writing data in {os.path.abspath(sql_path)}")
        engine = create_engine(f'sqlite:///{sql_path}')
        df.to_sql('table1', engine, index=False, if_exists='replace')
    return df


if __name__ == '__main__':
    if len(sys.argv) == 4:
        messages, categories, db_file = sys.argv[1 :]
        main(messages, categories, db_file)
    else:
        print("Please specify message .csv file, categories .csv file and output .db file")
