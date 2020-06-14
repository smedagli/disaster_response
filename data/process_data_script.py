"""
14/06/2020
Preprocessing of the data. Run the script as
will import messages and categories from the 2 .csv files, merge them and save into the .db file
See Also:
    data/process_data.py
"""
import argparse

from disaster_response import paths
from data.process_data import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--messages',
                      help=f'path of the .csv file containing the messages (default {paths.messages_path}',
                      default=paths.messages_path,
                      )
    parser.add_argument('-c', '--categories',
                      help=f'path of the .csv file containing the categories (default {paths.categories_path}',
                      default=paths.categories_path,
                      )
    parser.add_argument('-o', '--output',
                      help=f'path of the .db output file (default {paths.sql_path}',
                      default=paths.sql_path,
                      )

    args = parser.parse_args()
    main(messages_path=args.messages, category_path=args.categories, sql_path=args.output)
