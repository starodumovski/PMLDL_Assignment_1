import os
import wget
import zipfile
import pandas as pd
from tqdm import tqdm

CUR_FILE_FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT_FOLDER = os.path.join(CUR_FILE_FOLDER, '..', '..')

class LDataset:
    def __init__(self):
        self.load = {"from": "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip",
                     "to": os.path.join(ROOT_FOLDER, 'data', 'external')}
        self.unzip = {"from": os.path.join(ROOT_FOLDER, 'data', 'external', "filtered_paranmt.zip"),
                     "to": os.path.join(ROOT_FOLDER, 'data', 'raw')}

    def load_data(self):   
        print(self.load['to'])
        wget.download(self.load['from'], out=self.load['to'])

    def unzip_data(self):
        with zipfile.ZipFile(self.unzip['from'], 'r') as zip_ref:
            zip_ref.extractall(self.unzip['to'])

class PDataset:
    def __init__(self):
        self.raw_data = os.path.join(ROOT_FOLDER, 'data', 'raw', 'filtered.tsv')
        self.path_splitted = os.path.join(ROOT_FOLDER, 'data', 'interim')

    def logging(self, text):
        print(text)

    def read_data(self):
        return pd.read_csv(self.raw_data, sep='\t', index_col=0)

    def train_test_split(self, data, test_ratio=0.2):
        train_idx = int(len(data) * (1 - test_ratio))

        return data.iloc[:train_idx,:], data.iloc[train_idx:,:]
    
    def prepare_right_data(self, data):
        data_ref = data[data['ref_tox'] > data['trn_tox']]
        data_tra = data[~(data['ref_tox'] > data['trn_tox'])]
        
        data_to_return = pd.concat([
            pd.DataFrame(
                data_ref[['reference', 'translation', 'ref_tox', 'trn_tox', 'similarity']].values,
                columns=['toxic_sentence', 'neutral_sentence', 'toxic_tox', 'neutral_tox', 'similarity']
            ),
            pd.DataFrame(
                data_tra[['translation', 'reference', 'trn_tox', 'ref_tox', 'similarity']].values,
                columns=['toxic_sentence', 'neutral_sentence', 'toxic_tox', 'neutral_tox', 'similarity']
            )
        ], ignore_index=True)

        return data_to_return
    
    def split_files(self):
        self.logging("Reading the full data...")
        data = self.read_data()
        self.logging("Reconstructing the full data...")
        data = self.prepare_right_data(data)

        self.logging("Splitting the full data on test and train...")
        train_set, test_set = self.train_test_split(data)
        self.logging("Splitting the train data on val and train...")
        train_set, val_set = self.train_test_split(train_set)

        self.logging("Saving the train file...")
        train_set.to_csv(os.path.join(self.path_splitted, 'train.csv'), index=False, header=True, sep='\t')
        self.logging("Saving the test file...")
        test_set.to_csv(os.path.join(self.path_splitted, 'test.csv'), index=False, header=True, sep='\t')
        self.logging("Saving the validation file...")
        val_set.to_csv(os.path.join(self.path_splitted, 'val.csv'), index=False, header=True, sep='\t')

        self.logging("Ready")

if __name__ == '__main__':
    print('###Loading data step###')
    load_instance = LDataset()
    load_instance.load_data()
    load_instance.unzip_data()

    print('###Splitting data into samples step###')
    split_instance = PDataset()
    split_instance.split_files()