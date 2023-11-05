import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

CUR_FILE_FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT_FOLDER = os.path.join(CUR_FILE_FOLDER, '..', '..')

sys.path.insert(0, os.path.join(ROOT_FOLDER, 'src', 'data'))

from make_dataset import PDataset

class Visual:
    def __init__(self):
        self.to_save_path = os.path.join(ROOT_FOLDER, 'reports', 'figures')

    def save_picture(self, ax, ax_title, png_name, xlim=None):
        ax.set_title(ax_title)
        if xlim is not None:
            ax.set_xlim(*xlim)
        plt.savefig(os.path.join(self.to_save_path, f'{png_name}.png'))
        plt.close()
    
    def prepare_and_save_picture(self, data, column_name, ax_title, png_name):
        self.save_picture(
            ax=data[column_name].plot(kind='kde'),
            ax_title=ax_title,
            png_name=png_name,
            xlim=[min(data[column_name]), max(data[column_name])]
        )

    def read_data_save_pictures(self):
        print('Reading the full data...')
        data = PDataset().read_data()

        print('Rertieving the toxity of more and less toxic sentences separately...')
        data['max_tox'] = data[["ref_tox", "trn_tox"]].max(axis=1)
        data['min_tox'] = data[["ref_tox", "trn_tox"]].min(axis=1)

        print()

        print('Saving picture of similarity distribution...')
        self.prepare_and_save_picture(
            data=data,
            column_name='similarity',
            ax_title='Similarity distribution',
            png_name='similarity_distribution'
        )

        print('Saving picture of toxity of more toxic sentences from the pairs distribution...')
        self.prepare_and_save_picture(
            data=data,
            column_name='max_tox',
            ax_title='Distribution of toxity of more toxic sentences from the pairs',
            png_name='tox_tox_distribution'
        )
        print('Saving picture of toxity of less toxic sentences from the pairs distribution...')
        self.prepare_and_save_picture(
            data=data,
            column_name='min_tox',
            ax_title='Distribution of toxity of less toxic sentences from the pairs',
            png_name='neutral_tox_distribution'
        )

        print('Ready')

if __name__ == '__main__':
    print("### Vizualization step ###")
    viz = Visual()
    viz.read_data_save_pictures()