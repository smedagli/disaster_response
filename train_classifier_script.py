import argparse

from disaster_response import paths
from disaster_response.train_classifier import main

if __name__ == '__main__':
    parser=  argparse.ArgumentParser()
    parser.add_argument('-db', 'SQL_file',
                        help=f'path of the .db file containing messages and categories (default {paths.sql_path}',
                        default=paths.sql_path,
                        )
    parser.add_argument('-o', '--output',
                        help=f'file to save the model (default {paths.model_pickle_file}',
                        default=paths.model_pickle_file)

    args = parser.parse_args()
    main(database_file=args.SQL_file, pickle_file=args.output)
