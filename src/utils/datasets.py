import shutil
import tarfile
import zipfile

from os import makedirs, remove
from os.path import exists
from pandas import concat, DataFrame, merge, read_csv
from urllib.request import urlretrieve

from torchkge.data_structures import KnowledgeGraph

from torchkge.utils import get_data_home


def load_joke_dataset(data_path):
    train_path = data_path + '/train2id.txt'
    valid_path = data_path + '/valid2id.txt'
    test_path = data_path + '/test2id.txt'
    df1 = read_csv(train_path,
                   sep=' ', header=None, names=['from', 'rel', 'to'])
    if exists(valid_path):
        df2 = read_csv(valid_path,
                       sep=' ', header=None, names=['from', 'rel', 'to'])
    else:
        df2 = DataFrame([], columns=['from', 'rel', 'to'])
    if exists(test_path):
        df3 = read_csv(test_path,
                       sep=' ', header=None, names=['from', 'rel', 'to'])
    else:
        df3 = DataFrame([], columns=['from', 'rel', 'to'])
    df = concat([df1, df2, df3])
    kg = KnowledgeGraph(df)
    #kg.split_kg(sizes=(len(df1), len(df2), len(df3)))
    return kg
