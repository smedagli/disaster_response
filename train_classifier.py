from sqlalchemy import create_engine
import pandas as pd

from disaster_response import paths

def read_data():
    engine = create_engine(f'sqlite:///{paths.sql_path}')
    df = pd.read_sql('SELECT * FROM table1', con=engine)
    return df

df = read_data()
categ_col = [column for column in df.columns if column not in ['message', 'id', 'original', 'genre']]

X = df['message']
Y = df[categ_col]
