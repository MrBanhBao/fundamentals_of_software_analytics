import argparse
import git
import nltk
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


def main(output, dimred):
    # cwd = os.getcwd()
    cwd = '/Users/hao/workspace/hpi-de/2nd_Semester/fsa/exercise3/resources/webgl-operate'
    df = create_commit_df(cwd)
    vocab = create_vocab(df.tokens)
    df.to_hdf('alphaOnly.h5', key='commit')

    bow_df = create_bow_df(df, vocab)
    vecs = [np.array(list(bow.values())) for bow in bow_df.bow.values]

    if dimred == 'pca':
        pca = PCA(n_components=2)
        pca.fit(vecs)
        vecs_2d = pca.transform(vecs)
    else:
        tsne = TSNE(n_components=2)
        vecs_2d = tsne.fit_transform(vecs)

    pca_df = pd.DataFrame({'author.email': bow_df['author.email'],
                           'x': [x_y[0] for x_y in vecs_2d],
                           'y': [x_y[1] for x_y in vecs_2d]})

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='x', y='y', hue='author.email', data=pca_df)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig(output, bbox_inches="tight")


def create_commit_df(git_dir):
    repo = git.Repo(git_dir)
    data = [{
        'hexsha': commit.hexsha,
        'author.name': commit.author.name,
        'author.email': commit.author.email,
        'committed_date': pd.to_datetime(commit.committed_date, unit='s'),
        'commit_message': commit.message,
        'tokens': tokenize_diff(commit)
    } for commit in repo.iter_commits('--all')]

    return pd.DataFrame(data)


def tokenize_diff(commit, alpha_only=True):
    words = []

    parent = commit.parents[0] if commit.parents else None
    diffs = [diff for diff in commit.diff(parent, create_patch=True)]

    for diff in diffs:
        diff_text = diff.diff.decode("utf-8")
        tokens = nltk.word_tokenize(diff_text)
        if alpha_only:
            words += [word.lower() for word in tokens if word.isalpha()]
        else:
            words += [word.lower() for word in tokens]

    return words


def create_vocab(tokens, top=256):
    word_freqs = nltk.FreqDist(np.concatenate(tokens))
    top_words_freqs = word_freqs.most_common(top)
    top_words = [word_freq[0] for word_freq in top_words_freqs]
    return sorted(top_words)


def create_bow_df(df, vocab):
    bows = []
    author_emails = []
    for index, commit in df.iterrows():
        bow = get_empty_bow(vocab)
        author_email = commit['author.email']

        tokens = commit.tokens
        for token in tokens:
            if token in list(bow.keys()):
                bow[token] += 1

        author_emails.append(author_email)
        bows.append(bow)

    return pd.DataFrame({'author.email': author_emails, 'bow': bows})


def get_empty_bow(vocabs):
    word_vec = {}
    for vocab in vocabs:
        word_vec[vocab] = 0

    return word_vec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--output',
                        help='The file name of the output PDF file (default is output.pdf).',
                        default='./output.pdf',
                        required=False)

    parser.add_argument('-dr',
                        '--dimred',
                        help='Type of dimensionality reduction (pca or tsne).',
                        choices=['pca', 'tsne'],
                        default='pca',
                        required=False)

    args = parser.parse_args()
    main(**args.__dict__)
