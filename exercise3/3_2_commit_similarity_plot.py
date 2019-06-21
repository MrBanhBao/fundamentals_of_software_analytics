import argparse


def main(output):
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
