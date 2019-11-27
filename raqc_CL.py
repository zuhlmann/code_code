import argparse

# Basic Tutorial here: https://docs.python.org/2/howto/argparse.html

parser = argparse.ArgumentParser()
# Note since "masks" is not "--masks" it is a mandatory positional arg
parser.add_argument("mask", help="masks you wish to save")
parser.add_argument("-c", "--conditionals", help="placeholder if we want and/or functionality")
args = parser.parse_args()

# this will print out 'Running <name of this file>'
print('Running {}'.format(__file__))
