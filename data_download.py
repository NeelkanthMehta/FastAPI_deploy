import os
import wget

data_directory = 'data'
filename = 'data_banknote_authentication.txt'


if not filename in os.listdir(data_directory):
    wget.download(url='https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt', out=os.path.join(data_directory, filename))
