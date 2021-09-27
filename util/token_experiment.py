import csv
from pathlib import Path
from util import text_processing
from tensorflow_text import UnicodeCharTokenizer

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

tokens = [max(UnicodeCharTokenizer().tokenize(label).numpy()) for label in labels]

print(max(tokens))