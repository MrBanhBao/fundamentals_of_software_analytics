import argparse
import git
import numpy as np
import pandas as pd
import fnmatch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def main(output, pattern):
    # cwd = os.getcwd()
    cwd = '/Users/hao/workspace/hpi-de/2nd_Semester/fsa/exercise3/resources/webgl-operate'
    g = git.cmd.Git(cwd)

    df = create_commit_df(cwd)
    adj_matrix = create_adj_matrix(df)

    files = [file for file in g.ls_files().split('\n') if fnmatch.fnmatch(file, pattern)]

    filtered_adj_matrix = adj_matrix.loc[adj_matrix.index.str.contains('|'.join(files)),
                                         adj_matrix.columns.str.contains('|'.join(files))]

    plt.figure(figsize=(len(filtered_adj_matrix) / 3, len(filtered_adj_matrix) / 3))
    heatmap(filtered_adj_matrix.values, filtered_adj_matrix.index, filtered_adj_matrix.columns, cmap="YlGn")
    plt.savefig(output)


def create_commit_df(git_dir):
    repo = git.Repo(git_dir)
    data = [{
        'hexsha': commit.hexsha,
        'author.name': commit.author.name,
        'author.email': commit.author.email,
        'committed_date': pd.to_datetime(commit.committed_date, unit='s'),
        # 'commit_subject': commit.message.splitlines()[0],
        # 'commit_message': commit.message,
        'files': np.array(list(commit.stats.files.keys()))
    } for commit in repo.iter_commits('--all')]

    return pd.DataFrame(data)


def create_adj_matrix(df):
    file_names = np.unique(np.concatenate(df.files))
    adj_matrix = pd.DataFrame(np.zeros(shape=(len(file_names), len(file_names))), columns=file_names, index=file_names)

    for index, commit in df.iterrows():
        date = commit.committed_date
        start_date = (date - timedelta(days=3)).replace(hour=0, minute=0, second=0)
        end_date = (date + timedelta(days=3)).replace(hour=23, minute=59, second=59)
        author_email = commit['author.email']

        filtered_df = df[(df['committed_date'] >= start_date) & (df['committed_date'] <= end_date) & (
                df['author.email'] == author_email)]

        files_a = commit.files

        for file_a in files_a:
            for file_b in np.concatenate(filtered_df.files.values):
                if file_a != file_b:
                    adj_matrix[file_a][file_b] += 1

    return adj_matrix


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--output',
                        help='The file name of the output PDF file (default is output.pdf).',
                        default='./output.pdf',
                        required=False)

    parser.add_argument('-p',
                        '--pattern',
                        help='Unix filename pattern matching.',
                        default='*.ts',
                        required=False)

    args = parser.parse_args()
    main(**args.__dict__)
