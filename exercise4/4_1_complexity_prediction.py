import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def main(train, training_columns, predict, prediction_column):
    training_columns = [col.strip() for col in training_columns.split(';')]
    train_df = pd.read_csv(train, delimiter=';')

    X_t, y_t = preprocess_data(train_df, training_columns, prediction_column)
    reg = LinearRegression().fit(X_t, y_t)

    pred_df = pd.read_csv(predict, delimiter=';')
    predict_df(reg, pred_df, training_columns)


def preprocess_data(df, training_columns, prediction_column):
    X_t = df[training_columns].fillna(-1)
    y_t = df[prediction_column].fillna(-1)

    return X_t, y_t


def predict_df(reg, df, training_columns, prediction_column=None):
    for i, row in df.iterrows():
        X_p = row[training_columns].fillna(-1)
        pred_val = reg.predict([X_p])[0]

        if prediction_column:
            y_p = row[prediction_column]

            if np.isnan(y_p):
                y_p = -1
            print('{}; {}; {}'.format(row.filename, pred_val, y_p))
        else:
            print('{};{}'.format(row.filename, pred_val))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--train',
                        help='Path to the csv with the training data.',
                        required=True)

    parser.add_argument('-tc',
                        '--training-columns',
                        help='Features to use to learn prediction.',
                        required=True)

    parser.add_argument('-p',
                        '--predict',
                        help='Path to the csv with the prediction data.',
                        required=True)

    parser.add_argument('-pc',
                        '--prediction-column',
                        help='Column name with the expected output in csv passed with --predict.',
                        required=True)

    args = parser.parse_args()
    main(**args.__dict__)
