from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys
import os

from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast

CUR_FILE_FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT_FOLDER = os.path.join(CUR_FILE_FOLDER, '..', '..')

sys.path.insert(0, os.path.join(ROOT_FOLDER, 'src', 'data'))

from make_dataset import MyDataset


if __name__ == "__main__":
    print('Loading the model...')
    # model = PegasusForConditionalGeneration.from_pretrained(os.path.join(ROOT_FOLDER, 'models', 'model.h5'))
    model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
    print('Loading the tokenizer...')
    tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")

    def collate_fn(batch):
        '''function to proccess the batch before passing to the model
        described in report_2 (Training process section)
        '''
        batch = np.array(batch)
        toxic, neutral = batch[:, 0], batch[:, 1]
        toxic = tokenizer.batch_encode_plus(toxic.tolist(), add_special_tokens=True,
                                            padding='max_length', return_tensors='pt')['input_ids']
        
        model_gen = model.generate(toxic, temperature=1.0)
        
        neutral = tokenizer.batch_encode_plus(neutral.tolist(), add_special_tokens=True,
                                            truncation=True,
                                            padding='max_length', return_tensors='pt',
                                            max_length=len(model_gen[0]))['input_ids']
        return toxic, neutral, model_gen

    def train_loop(model, dataloader, num_epoch=1, smaple_size=2):
        '''train function'''
        model.train()
        pbar = tqdm(enumerate(dataloader), total=smaple_size)
        for epoch in range(num_epoch):
            losses = []
            for idx, batch in pbar:
                if idx == smaple_size:
                    break
                
                toxic, neutral, model_gen = batch
                loss = model(input_ids=toxic, decoder_input_ids=model_gen, labels=neutral)[0]
                losses.append(loss.item())
                loss.backward()
                
                pbar.set_postfix({
                    'Epoch': f'{epoch + 1}/{num_epoch}',
                    'Batch': f'{idx + 1}/{smaple_size}',
                    'Loss': f"{sum(losses) / len(losses)}"
                })
        model.eval()

    print('Dataset creation...')
    dataset = MyDataset()
    test_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    print('Training process...')
    train_loop(model, test_loader, smaple_size=10)

    print('Saving the model...')
    model.save_pretrained(os.path.join(ROOT_FOLDER, 'models', 'model.h5'))

