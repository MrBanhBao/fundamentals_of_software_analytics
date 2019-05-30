import os


# cwd = './qtbase'
cwd = os.getcwd()


def count_lines(file):
    with open(file) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


for dir_path, dir_names, file_names in os.walk(cwd):
    for file_name in file_names:
        try:
            lines = count_lines(os.path.join(dir_path, file_name))
            print('{};{}'.format(file_name, lines))
        except:
            continue

