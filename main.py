import dynet as dy
from datetime import datetime
import random
import os
import sys


WORD_EMBED_SIZE = 300



# args[1] => use pretrained

def get_vocab():


def main():
    if sys.argv[1] == 1:
        return None
    else:
        vocab = get_vocab()
    # rare_words = get_rare_words()
    # divide into batches
    # get vocab
    # add vocab - unknown word

    # create word2index
    # create label2index

    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    # create class holder
    # create all LSTMs and MLPs and put it into holder

    start_training_time = datetime.now()
    current_date = start_training_time.strftime("%d.%m.%Y")
    current_time = start_training_time.strftime("%H:%M:%S")

    # start training + evaluation

    # save model


if __name__ == '__main__':
    main()