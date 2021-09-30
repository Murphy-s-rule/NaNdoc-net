import csv
from pathlib import Path
from util import text_processing
import tensorflow as tf
from datasets import dataset

path = Path('datasets/snukb/dataset')
train_path = path / 'train'
train_label_path = train_path / 'train.csv'
test_path = path / 'test'
test_label_path = test_path / 'test.csv'
with train_label_path.open() as f:
    train_labels = [row['label'] for row in csv.DictReader(f)]
with train_label_path.open() as f:
    test_labels = [row['label'] for row in csv.DictReader(f)]


labels = train_labels + test_labels

tokens = [text_processing.HangulTokenizer().tokenize(label,58) for label in labels]


for t in tokens:
    print(t)

print(dataset.get_max_len_label_tokens(train_label_path))

