"""datasets datasets."""
import tensorflow_datasets as tfds
from pathlib import Path
import csv
from util import text_processing

# TODO(datasets): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(datasets): BibTeX citation
_CITATION = """
"""


class Snukb(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for datasets datasets."""

    VERSION = tfds.core.Version('1.0.1')
    RELEASE_NOTES = {
        '1.0.1': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the datasets metadata."""
        # TODO(datasets): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your datasets like images, labels ...
                'image_describe': tfds.features.Text(),
                'image': tfds.features.Image(),
                'label': tfds.features.Text(),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(datasets): Downloads the data and defines the splits
        path = Path('datasets/snukb/dataset')

        # TODO(datasets): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(path / 'train'),
            'test': self._generate_examples(path / 'test'),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(datasets): Yields (key, example) tuples from the datasets

        label_path = path / f'{path.parts[-1]}.csv'

        with label_path.open() as f:
            for row in csv.DictReader(f):
                image_id = row['index']
                print(row['label'])
                yield image_id, {
                    'image_describe': '',
                    'image': path / 'images' / f'{image_id}.jpg',
                    'label': row['label']
                }