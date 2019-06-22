import os
import argparse
import nltk
import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import seaborn as sns
import matplotlib.pyplot as plt


def main(output):
    # cwd = os.getcwd()
    cwd = '/Users/hao/workspace/hpi-de/2nd_Semester/fsa/exercise3/resources/webgl-operate/'

    df = create_df(cwd)
    vocab = create_vocab(df.tokens)

    bow_df = create_bow_df(df, vocab)
    vecs = [np.array(list(bow.values())) for bow in bow_df.bow.values]

    mds = MDS(n_components=2)
    vecs_2d = mds.fit_transform(vecs)

    mds_df = pd.DataFrame({'file': bow_df.file,
                           'x': [x_y[0] for x_y in vecs_2d],
                           'y': [x_y[1] for x_y in vecs_2d]})

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='x', y='y', data=mds_df, legend=False)
    label_point(mds_df.x, mds_df.y, mds_df.file, plt.gca())
    plt.savefig(output)


def create_df(cwd):
    file_list = []
    word_list = []

    for root, dirs, files in os.walk(cwd):
        for file in files:
            path = os.path.join(root, file)

            try:
                with open(path) as f:
                    source_code = f.read()
                    words = tokenize_source_code(source_code)

                    file_list.append(file)
                    word_list.append(words)
            except Exception as e:
                # print(e, 'at file', file)
                continue

    return pd.DataFrame({'file': file_list, 'tokens': word_list})


def tokenize_source_code(source_code, alpha_only=True):
    tokens = nltk.word_tokenize(source_code)

    if alpha_only:
        return [word.lower() for word in tokens if word.isalpha()]
    else:
        return [word.lower() for word in tokens]


def create_vocab(tokens, top=256):
    word_freqs = nltk.FreqDist(np.concatenate(tokens))
    top_words_freqs = word_freqs.most_common(top)
    top_words = [word_freq[0] for word_freq in top_words_freqs]
    return sorted(top_words)


def create_bow_df(df, vocab):
    bows = []
    files = []
    for index, row in df.iterrows():
        bow = get_empty_bow(vocab)
        file = row.file

        tokens = row.tokens
        for token in tokens:
            if token in list(bow.keys()):
                bow[token] += 1

        files.append(file)
        bows.append(bow)

    return pd.DataFrame({'file': files, 'bow': bows})


def get_empty_bow(vocabs):
    word_vec = {}
    for vocab in vocabs:
        word_vec[vocab] = 0

    return word_vec


def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--output',
                        help='The file name of the output PDF file (default is output.pdf).',
                        default='./output.pdf',
                        required=False)

    args = parser.parse_args()
    main(**args.__dict__)

