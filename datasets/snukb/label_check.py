import csv
from pathlib import Path

def check_longest_label(label_path):
    with label_path.open() as f:
        label_lens = [len(row['label']) for row in csv.DictReader(f)]
        return max(label_lens)

if __name__ == '__main__':
    dataset_path = Path('datasets/snukb/dataset')
    train_max = check_longest_label(dataset_path / 'train/train.csv')
    test_max = check_longest_label(dataset_path / 'test/test.csv')
    print(max(train_max,test_max))