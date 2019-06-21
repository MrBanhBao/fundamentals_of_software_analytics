import argparse
import os

def main(output):
    # cwd = os.getcwd()
    cwd = '/home/hao/mystuff/fundamentals_of_software_analytics/exercise3/resources/webgl-operate'
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--output',
                        help='The file name of the output PDF file (default is output.pdf).',
                        default='./output.pdf',
                        required=False)


    args = parser.parse_args()
    main(**args.__dict__)
