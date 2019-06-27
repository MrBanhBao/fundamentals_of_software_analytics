import sys
import git
import os
import pandas as pd
import numpy as np


def main():
    # cwd = os.getcwd()
    cwd = '/Users/hao/workspace/hpi-de/2nd_Semester/fsa/exercise3/resources/uBlock'
    repo = git.Repo(cwd)

    authors_df = create_authors_df(repo)
    modules_df = create_modules_df(repo)

    agg_authors_df = calc_aggr_author_df(authors_df)
    agg_modules_df = calc_aggr_module_df(modules_df)
    agg_edges_df = agg_edge_df(authors_df)

    print('hierarchy;authors;{}'.format(len(agg_authors_df)))
    for index, row in agg_authors_df.iterrows():
        print('node;author;{};{};{}'.format(row['author.email'], row.noc, row.mtbc))

    print('hierarchy;modules;{}'.format(len(agg_modules_df)))
    for index, row in agg_modules_df.iterrows():
        print('node;module;{};{};{}'.format(row.module, row.noc, row.mtbc))

    print('edges;edits;{}'.format(len(agg_edges_df)))
    for index, row in agg_edges_df.iterrows():
        print('edge;edit;{};{}'.format(row['author.email'], row.module))


def create_authors_df(repo):
    data = [{
        'hexsha': commit.hexsha,
        'author.email': commit.author.email,
        'committed_date': pd.to_datetime(commit.committed_date, unit='s'),
        'num_changes': len(list(commit.stats.files.keys())),
        'files': list(commit.stats.files.keys())
    } for commit in repo.iter_commits('--all')]

    return pd.DataFrame(data)


def create_modules_df(repo):
    file_list = []
    author_emails = []
    commited_dates = []
    hexshas = []

    for commit in repo.iter_commits('--all'):
        files = list(commit.stats.files.keys())
        for file in files:
            file_list.append(file)
            author_emails.append(commit.author.email)
            commited_dates.append(pd.to_datetime(commit.committed_date, unit='s'))
            hexshas.append(commit.hexsha)

    return pd.DataFrame({'file': file_list, 'author.email': author_emails,
                         'committed_date': commited_dates, 'hexsha': hexshas}).fillna(0)


def calc_aggr_author_df(author_df):
    authors = []
    nocs = []
    mtbcs = []
    authors_emails = author_df['author.email'].unique()

    for author_email in authors_emails:
        df = author_df[author_df['author.email'] == author_email]

        noc = df.num_changes.sum()
        diffs = df.committed_date.sort_values().diff() / np.timedelta64(1, 'D')
        mtbc = diffs.mean()

        authors.append(author_email)
        nocs.append(noc)
        mtbcs.append(mtbc)

    return pd.DataFrame({'author.email': authors, 'noc': nocs, 'mtbc': mtbcs}).fillna(0)


def calc_aggr_module_df(module_df):
    modules = []
    nocs = []
    mtbcs = []

    files = module_df.file
    for file in files:
        df = module_df[module_df.file == file]

        noc = len(df)
        diffs = df.committed_date.sort_values().diff() / np.timedelta64(1, 'D')
        mtbc = diffs.mean()

        modules.append(file)
        nocs.append(noc)
        mtbcs.append(mtbc)

    return pd.DataFrame({'module': modules, 'noc': nocs, 'mtbc': mtbcs})


def agg_edge_df(author_df):
    authors = []
    files = []
    for index, row in author_df.iterrows():
        for file in row.files:
            authors.append(row['author.email'])
            files.append(file)

    return pd.DataFrame({'author.email': authors, 'module': files})


if __name__ == '__main__':
    main()
