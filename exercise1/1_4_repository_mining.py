import os
import git
import re
import operator
import datetime
import time
import numpy as np
from dateutil.relativedelta import relativedelta
import pandas as pd

# cwd = './qtbase'
cwd = os.getcwd()

repo = git.Repo(cwd)
data = [{
        'hexsha': commit.hexsha,
        'author.name': commit.author.name,
        'author.email': commit.author.email,
        'committed_date': pd.to_datetime(commit.committed_date, unit='s'),
        'commit_subject': commit.message.splitlines()[0],
        'commit_message': commit.message,
        'parents': commit.parents
        } for commit in repo.iter_commits('5.12')]
df = pd.DataFrame(data)

# Number of authors
print('Number of authors:', len(df['author.email'].unique()))


# Most active author within last year
last_year_df = df[df['committed_date'] >= datetime.datetime.now() - pd.to_timedelta('365day')]
most_active_author_name = last_year_df.groupby('author.name').size().sort_values(ascending=False).index[0]
most_active_author_email = last_year_df[last_year_df['author.name'] == most_active_author_name].iloc[0]['author.email']

print('Most active author within last year: {} <{}>'.format(most_active_author_name, most_active_author_email))


# Time range of development
def __create_time_str(diff, unit):
    unit_str = ''
    if getattr(diff, unit) >= 1:
        if getattr(diff, unit) == 1:
            unit_str = '{} {}'.format(getattr(diff, unit), unit[:-1])
        else:
            unit_str = '{} {}'.format(getattr(diff, unit), unit)

    return unit_str


def pretty_print_date(diff, units=['years', 'months', 'days']):
    out_str = '('
    for i, unit in enumerate(units):
        unit_str = __create_time_str(diff, unit)
        if unit_str:
            if i != len(units) - 1:
                out_str = out_str + unit_str + ', '
            else:
                out_str = out_str + unit_str + ' ago)'
    return out_str


min_time = df['committed_date'].min()
max_time = pd.to_datetime(int(time.time()), unit='s')  # df['committed_date'].max()

start = datetime.datetime.strptime(str(min_time), '%Y-%m-%d %H:%M:%S')
end = datetime.datetime.strptime(str(max_time), '%Y-%m-%d %H:%M:%S')

diff = relativedelta(end, start)

print('Time range of development: {} days {}'.format((max_time - min_time).days, pretty_print_date(diff)))


# Share of maintenance
search_keywords = ['fix', 'rewrite', 'doc', 'test', 'improve']
no_merge_commits = df[df['parents'].apply(len) == 1]
df_filtered = no_merge_commits[((no_merge_commits['commit_message'].str.lower().str.contains('|'.join(search_keywords))) &
                              ~(no_merge_commits['commit_message'].str.lower().str.contains('cherry')))]

share_of_maintenance = 100/len(df)*len(df_filtered)

print('Share of maintenance: {0:.1f}%'.format(share_of_maintenance))


# Share of stale code
def get_changed_files(commit_a, commit_b):
    changed_files = []
    for x in commit_a.diff(commit_b):
        if x.a_path is not None and x.a_path not in changed_files:
            changed_files.append(x.a_path)
        elif x.b_path is not None and x.b_path not in changed_files:
            changed_files.append(x.b_path)
    return changed_files


last_year_df = df[df['committed_date'] >= datetime.datetime.now() - pd.to_timedelta('365day')]
sha_a = last_year_df.iloc[0].hexsha
sha_b = last_year_df.iloc[-1].hexsha
commit_a = repo.commit(sha_a)
commit_b = repo.commit(sha_b)

changed_files = get_changed_files(commit_a, commit_b)

was_changed = []
for dir_path, dir_names, file_names in os.walk(cwd):
    for file_name in file_names:
        try:
            if cwd[-1] != '/':
                cwd = cwd + '/'

            path = os.path.join(dir_path, file_name).replace(cwd, '')

            if path in changed_files:
                was_changed.append(True)
            else:
                was_changed.append(False)
        except:
            continue

print(f'Share of stale code: {100 - (100/len(was_changed) * np.array(was_changed).sum()):.1f}%')


# Top 10 commit message keywords within last month
def clean_text_tokenize(text,
                        excluded_words=['a', 'an', 'at', 'for', 'from', 'in', 'is', 'of', 'on', 'the', 'to', 'use',
                                        'with', 'when']):
    text = text.lower()

    text = re.sub(r'[^\w |#]', '', text)
    resultwords = [word for word in text.split(' ') if word not in excluded_words]
    return resultwords


def get_top_keywords(df, n=10):
    word_freq = {}
    for commit_subject in df.commit_subject:
        tokens = clean_text_tokenize(commit_subject)
        for token in tokens:
            if token:
                if token in word_freq:
                    word_freq[token] = word_freq[token] + 1
                else:
                    word_freq[token] = 1

    word_freq_s = sorted(word_freq.items(), key=operator.itemgetter(1), reverse=True)

    return [tup[0] for tup in word_freq_s[:n]]


top_keywords = get_top_keywords(df)

print('Top 10 commit message keywords last month: {}'.format(re.sub(r'[\'|\[\]]', '', str(top_keywords))))
