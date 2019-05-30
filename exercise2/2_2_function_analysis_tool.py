import argparse
import os
import sys
import ast
import pandas as pd
import seaborn as sns


def main(output, type):
    cwd = os.getcwd()
    func_params = []
    body_locs = []
    for stdin in sys.stdin:
        file = str.strip(stdin)

        with open(os.path.join(cwd, file)) as f:
            source_code = f.read()
            try:
                tree = ast.parse(source_code)
            except Exception as e:
                print('Error:', e, 'at', file)
                continue

            source_code_lines = source_code.split('\n')

            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

            for function in functions:
                if is_indentation_level(function, level=0):
                    num_params = get_num_of_parameters(function)
                    func_params.append(num_params)

                    num_loc = get_body_lines_of_func(function, source_code_lines)
                    body_locs.append(num_loc)

    df = create_pd_df(func_params, body_locs)

    if type == 'box':
        plot_box(df, output)
    else:
        plot_dot(df, output)

    print('Done.')


def is_indentation_level(function_node, level=0):
    return function_node.col_offset == level


def get_body_lines_of_func(function_node, source_code_lines):
    counter = 0
    function_line_start = function_node.lineno

    for line in source_code_lines[function_line_start:]:
        if line is not '':
            counter += 1
        else:
            return counter


def get_num_of_parameters(function_node):
    return len(function_node.args.args)


def create_pd_df(x, y):
    return pd.DataFrame({'Number of Parameters': x, 'Lines of Codes': y})


def plot_box(data, path, x='Number of Parameters', y='Lines of Codes'):
    plot = sns.boxplot(x=x, y=y, data=data)
    plot.figure.savefig(path)


def plot_dot(data, path, x='Number of Parameters', y='Lines of Codes'):
    plot = sns.scatterplot(x=x, y=y, data=data)
    plot.figure.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--output',
                        help='The file name of the output PDF file (default is output.pdf).',
                        default='./output.pdf',
                        required=False)

    parser.add_argument('-t',
                        '--type',
                        help='Define the plot type (either box or dot, default is dot).',
                        choices=['box', 'dot'],
                        default='dot',
                        required=False)

    args = parser.parse_args()
    main(**args.__dict__)
