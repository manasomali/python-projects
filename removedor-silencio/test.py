from os import path
import glob

directory = path.dirname(path.realpath(__file__))
directory = path.dirname(path.realpath(__file__)) + '\input\*.wav'
caminhos = (glob.glob(directory))