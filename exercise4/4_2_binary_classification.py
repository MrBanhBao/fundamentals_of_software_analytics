import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def main(train, predict, type, output_error_values):
    train_df = parse_to_df(train)
    pred_df = parse_to_df(predict)

    X_t, y_t = preprocess_data(train_df)

    if type == 'support-vector-machine':
        model = SVC(kernel='rbf', gamma=2, C=1.0).fit(X_t, y_t)
    else:
        model = LogisticRegression().fit(X_t, y_t)

    X_p, y_p = preprocess_data(pred_df)

    if output_error_values:
        train_score = model.score(X_t, y_t)
        pred_score = model.score(X_p, y_p)

        err_report = create_error_report(type, train_score, pred_score)

        with open("nasa_error_report.txt", "w") as text_file:
            text_file.write(err_report)
            text_file.close()
        print(err_report)

    else:
        y_o = model.predict(X_p)
        for val in y_o:
            print(val)
            if val:
                print('Y')
            else:
                print('N')


def parse_to_df(file):
    with open(file) as f:
        mode = ''
        col_names = []
        col_types = []
        data = []
        for line in f.readlines():
            if '@relation' in line:
                mode = '@relation'
            elif '@data' in line:
                mode = '@data'

            if mode == '@relation':
                line_splits = line.rstrip().split(' ')
                if len(line_splits) == 3:
                    col_name = line_splits[1]
                    col_type = line_splits[2]

                    col_names.append(col_name)
                    col_types.append(col_type)

            if mode == '@data':
                line_splits = line.rstrip().split(',')
                cast_vals = []
                if len(line_splits) > 1:
                    for idx, val in enumerate(line_splits):
                        cast_vals.append(cast(val, col_types[idx]))

                    data.append(tuple(cast_vals))

    return pd.DataFrame(data, columns=col_names)


def cast(val, col_type):
    if val == '?':
        return np.nan
    elif col_type == '{Y,N}':
        return True if val == 'Y' else False
    elif col_type == 'numeric':
        return float(val)
    else:
        return val


def preprocess_data(df):
    X = df.iloc[:, 0:-1].fillna(-1)
    y = df.iloc[:, -1]

    mask = ~y.isnull()

    return X[mask], y[mask]


def create_error_report(type, train_score, pred_score):
    err_report = ''
    if type == 'logistic-regression':
        err_report += 'Logistic regression error report\n'
    else:
        err_report += 'Logistic regression error report\n'

    err_report += 'train error: {}%\n'.format((1 - train_score)*100)
    err_report += 'prediction error: {}%\n'.format((1 - pred_score)*100)

    return err_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--train',
                        help='Path to the csv with the training data.',
                        required=True)

    parser.add_argument('-p',
                        '--predict',
                        help='Path to the csv with the prediction data.',
                        required=True)

    parser.add_argument('-ty',
                        '--type',
                        help='Choose model type.',
                        choices=['logistic-regression', 'support-vector-machine'],
                        default='logistic-regression',
                        required=False)

    parser.add_argument('-oev',
                        '--output-error-values',
                        help='Print the full file path',
                        action='store_true',
                        required=False,
                        default=False)

    args = parser.parse_args()
    main(**args.__dict__)
