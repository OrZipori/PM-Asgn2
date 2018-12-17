# Or Zipori 302933833
# Shauli Ravfogel 308046861
import sys
import utils
import create_output

def main():
    args = sys.argv
    dev_fname, test_fname, input_word, output_fname = sys.argv[1:]
    #dev_fname, test_fname = './dataset/develop.txt', './dataset/test.txt'
    create_output.create( dev_fname, test_fname, output_fname, input_word)

if __name__ == "__main__":
    main()
