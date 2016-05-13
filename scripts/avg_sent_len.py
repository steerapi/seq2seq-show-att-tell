import sys
import numpy as np
import argparse


def get_avg_words(filename):
    num_sentences = 0
    num_words = 0
    f = open(filename)
    
    for line in f:
        num_sentences += 1
        words = line.split(' ')
        num_words += len(words)
        
    print num_words / float(num_sentences)
    
    return


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-filename', help="File name",
                        type=str)
    args = parser.parse_args(arguments)
    filename = args.filename
    get_avg_words(filename)
    return


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


