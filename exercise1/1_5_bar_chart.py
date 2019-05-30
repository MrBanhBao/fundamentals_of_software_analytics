import argparse
import math
import pandas as pd
import sys


def main(attribute, sort, limit, hierarchical, flat, columns, key):
    df = pd.read_csv(sys.stdin, delimiter=';')
    asc = False
    if not attribute:
        attribute = df.columns[1]

    if sort == 'asc':
        asc = True

    if limit is None:
        limit = len(df)

    df_attr = df[[key, attribute]].dropna()
    df_attr = df_attr.sort_values(by=attribute, ascending=asc)[:limit]

    if flat:
        df_attr[key] = df_attr[key].apply(lambda filename: filename.split('/')[-1])

    max_key = df_attr[key].str.len().max()
    max_val = df_attr[attribute].max()
    rest_columns = columns - max_key - 3  # -3 because of _|_ (_ is space)

    for index, row in df_attr.iterrows():
        key_value = row[key]
        attr_value = row[attribute]

        num_pluses = rest_columns/max_val * attr_value
        if math.isnan(num_pluses):
            num_pluses = 0
        else:
            num_pluses = int(num_pluses)

        pluses = '+' * num_pluses
        print(f'{key_value.rjust(max_key)} ▏{pluses}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a',
                        '--attribute',
                        help='allow to select the attribute. '
                             'If no attributeis selected, use the first one after the key.',
                        required=False)
    parser.add_argument('-s',
                        '--sort',
                        help='Allow options to sort in ascending order and descending order.',
                        choices=['desc', 'asc'],
                        default='asc',
                        required=False)
    parser.add_argument('-l',
                        '--limit',
                        type=int,
                        help='Allow to limit the number of lines of the output',
                        required=False)
    parser.add_argument('-hi',
                        '--hierarchical',
                        help='Print the full file path',
                        action='store_true',
                        required=False,
                        default=False)
    parser.add_argument('-f',
                        '--flat',
                        help='print only the file name part of the file path',
                        action='store_true',
                        required=False,
                        default=True)
    parser.add_argument('-c',
                        '--columns',
                        help='column width for the output',
                        type=int,
                        required=False,
                        default=80)
    parser.add_argument('-k',
                        '--key',
                        help='key column of csv',
                        type=str,
                        default='filename',
                        required=False)

    args = parser.parse_args()
    main(**args.__dict__)
