import sys
import os

from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast

CUR_FILE_FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT_FOLDER = os.path.join(CUR_FILE_FOLDER, '..', '..')

sys.path.insert(0, os.path.join(ROOT_FOLDER, 'src', 'data'))

from make_dataset import MyDataset


if __name__ == "__main__":
    # Loading the model
    model = PegasusForConditionalGeneration.from_pretrained(os.path.join(ROOT_FOLDER, 'models', 'model.h5'))
    # model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
    tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")

    dataset = MyDataset()

    print(f"Initial sentence: {dataset[4][0]}")

    # prediction of the final sentence
    final_sentence = tokenizer.batch_decode(
        model.generate(
        tokenizer.encode(dataset[4][0], return_tensors='pt'),
        temperature=1.0
    ),
    skip_special_tokens=True)
    print(f"Paraphrased sentence: {final_sentence}")
