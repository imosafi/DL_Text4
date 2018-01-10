import dynet as dy
import datetime
import random
import os

EMBED_SIZE = 300




def main():
    model = dy.Model()
    trainer = dy.AdamTrainer(model)
    embeds = model.add_lookup_parameters((VOCAB_SIZE, EMBED_SIZE))



if __name__ == '__main__':
    main()