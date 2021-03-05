from os import path
import os

outputdirectory = path.dirname(path.realpath(__file__)) + '\output'

for path, subdirs, files in os.walk(outputdirectory):
    for name in files:
        print(os.path.join(path, name))