import git
import os
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path


def main():
    cwd = os.getcwd()
    # cwd = '/Users/hao/workspace/hpi-de/2nd_Semester/fsa/exercise2/Vulkan-Examples/'

    g = git.cmd.Git(cwd)
    repo_path = get_git_repo_path(cwd)
    usr_path = str(Path(repo_path).parents[0])
    repo = git.Repo(repo_path)

    src_obj_dict = create_src_obj_dict(os.path.join(repo_path, 'compile_commands.json'), usr_path)

    print('# filename;MTBC;NoC;BF;OSpLoC;SoVkC')
    for git_file in g.ls_files().split('\n'):
        if is_c_file(git_file):
            commits = repo.iter_commits('--all', since='365.days.ago', paths=git_file)

            data = []
            for commit in commits:
                data.append({'filename': git_file,
                             'commit.hash': commit.hexsha,
                             'author.email': commit.author.email,
                             'committed_date': pd.to_datetime(commit.committed_date, unit='s')})

            if len(data) > 0:
                df = pd.DataFrame(data)

                mtcb = calc_MTCB(df)
                noc = calc_NoC(df)
                bf = calc_BF(df)
                osploc = calc_OSoLoC(src_obj_dict, repo_path, git_file)
                sovkc = calc_SoVkC(repo_path, git_file)

                print('{};{};{};{:0.2f};{:0.2f};{:0.2f}'.format(git_file, mtcb, noc, bf, osploc, sovkc))


def get_git_repo_path(cwd):
    if is_git_repo(cwd):
        return cwd
    else:
        num_parents = cwd.count('/')
        for parent_idx in range(0, num_parents):
            parent_path = str(Path(cwd).parents[parent_idx])

            if is_git_repo(parent_path):
                return parent_path


def is_git_repo(path):
    dirs = os.listdir(path)
    return '.git' in dirs


def create_src_obj_dict(json_file, user_path):
    source_obj_dict = {}
    with open(json_file, 'r') as f:
        datastore = json.load(f)

        for data in datastore:
            obj_file_dir = data['directory'].replace('/root/', '')
            obj_file_path = data['command'].split(' ')[-3]
            source_file_path = data['command'].split(' ')[-1]

            source_file_path = '/'.join(source_file_path.split('/')[3:])
            obj_file_path = os.path.join(user_path, obj_file_dir, obj_file_path)

            source_obj_dict[source_file_path] = obj_file_path

    return source_obj_dict


def is_c_file(file):
    return file.lower().endswith(('.h', '.hpp', '.c', '.cpp'))


# Mean Time Between Changes
def calc_MTCB(df):
    time_diffs = df['committed_date'][::-1].diff()
    return np.mean(time_diffs).days


# Number of Collaborators
def calc_NoC(df):
    return len(df['author.email'].unique())


# Botch Factor
def calc_BF(df):
    mtcb = calc_MTCB(df)
    noc = calc_NoC(df)
    if mtcb is not 0:
        return np.power(noc, 2) / mtcb
    else:
        return 0


def calc_OSoLoC(source_obj_dict, repo_path, git_file):
    src_loc = get_LoC(os.path.join(repo_path, git_file))
    if git_file in source_obj_dict.keys():
        obj_size = get_obj_size(source_obj_dict[git_file])
        return obj_size/src_loc
    else:
        return 0


def get_LoC(file):
    with open(file, 'rb') as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def calc_SoVkC(repo_path, git_file):
    try:
        with open(os.path.join(repo_path, git_file)) as f:
            content = f.read()

            symbols = re.findall(r'([\d\w_]+)', content.lower())
            vk_symbols = [symbol for symbol in symbols if 'vk' in symbol]

            return len(symbols) / len(vk_symbols)
    except:
        return 0


def get_obj_size(file):
    return os.path.getsize(file)


if __name__ == '__main__':
    main()




