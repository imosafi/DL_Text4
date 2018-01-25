import dynet as dy
from datetime import datetime
import random
import os
import sys
from enum import Enum
from collections import Counter
import numpy as np

SNLI_VOCAB_SIZE = 40000
BATCH_SIZE = 40
WORD_EMBED_SIZE = 300
EPOCHS = 5
EVALUATE_ITERATION = 800

method = 'larger_layers_no_dropout'

dev_evaluation_file = 'dev_eval.txt'
test_evaluation_file = 'test_eval.txt'

CHATS_TO_SKIP = ',?'
SEPERATION_STR = '$$$'
# UNKNOWN_WORD = 'UnKnOwN_wOrD'
UNKNOWN_WORD = 'UNK'

labelDict = {'neutral': 0, 'entailment': 1, 'contradiction': 2, '-': 3}

USE_PRETRINED = True


class TrainingOn(Enum):
    snli = 1
    multinli = 2
    both = 3


def take_labels_0_1_2(trainset):
    return [t for t in trainset if t[2] != 3]


def get_rare_words(words, vocab_size):
    counter = Counter(words)
    most_common_keys = [x[0] for x in counter.most_common(vocab_size)]
    keys_to_remove = [x for x in counter.keys() if x not in most_common_keys]
    return keys_to_remove


def split_into_batches(set, size):
    return [set[i:i + size] for i in range(0, len(set), size)]


def get_set(file_name):
    trainset = []
    vocab = []
    for line in file(file_name):
        line = line.strip(' \t\n')
        parts = line.split(SEPERATION_STR)
        if not len(parts) == 3:
            raise Exception("more than 3 parts")
        parts[0] = parts[0].strip('. \t').split()
        parts[1] = parts[1].strip('. \t').split()
        parts[2] = parts[2].strip('. \t')
        parts[0] = [x for x in parts[0] if x not in list(CHATS_TO_SKIP)]
        parts[1] = [x for x in parts[1] if x not in list(CHATS_TO_SKIP)]
        for word in parts[0]:
            vocab.append(word)
        for word in parts[1]:
            vocab.append(word)
        trainset.append([parts[0], parts[1], int(parts[2])])
    trainset = take_labels_0_1_2(trainset)
    return vocab, trainset


def get_word_rep(word, holder):
    w_index = holder.word2index[word] if word in holder.word2index else holder.word2index[UNKNOWN_WORD]
    return holder.word_embedding[w_index]


class ComponentHolder:
    def __init__(self, word2index, fwdRNN_layer1, bwdRNN_layer1, fwdRNN_layer2, bwdRNN_layer2, fwdRNN_layer3, bwdRNN_layer3,
                 W1, b1, W2, b2, word_embedding):
        self.word2index = word2index
        self.fwdRNN_layer1 = fwdRNN_layer1
        self.bwdRNN_layer1 = bwdRNN_layer1
        self.fwdRNN_layer2 = fwdRNN_layer2
        self.bwdRNN_layer2 = bwdRNN_layer2
        self.fwdRNN_layer3 = fwdRNN_layer3
        self.bwdRNN_layer3 = bwdRNN_layer3
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.word_embedding = word_embedding


def build_graph(pre_words, hy_words, holder):
    fl1_init = holder.fwdRNN_layer1.initial_state()
    bl1_init = holder.bwdRNN_layer1.initial_state()
    fl2_init = holder.fwdRNN_layer2.initial_state()
    bl2_init = holder.bwdRNN_layer2.initial_state()
    fl3_init = holder.fwdRNN_layer3.initial_state()
    bl3_init = holder.bwdRNN_layer3.initial_state()

    pre_wembs = [get_word_rep(w, holder) for w in pre_words]
    hy_wembs = [get_word_rep(w, holder) for w in hy_words]


    pre_fws = fl1_init.transduce(pre_wembs)
    pre_bws = bl1_init.transduce(reversed(pre_wembs))

    pre_bi = [dy.concatenate([word, f, b]) for word, f, b in zip(pre_wembs, pre_fws, reversed(pre_bws))]
    pre_fws2 = fl2_init.transduce(pre_bi)
    pre_bws2 = bl2_init.transduce(reversed(pre_bi))

    pre_b_tag = [dy.concatenate([word, f1, b1, f2, b2]) for word, f1, b1, f2, b2 in zip(pre_wembs, pre_fws, reversed(pre_bws), pre_fws2, reversed(pre_bws2))]
    pre_fws3 = fl3_init.transduce(pre_b_tag)
    pre_bws3 = bl3_init.transduce(reversed(pre_b_tag))

    pre_b_tagtag = [dy.concatenate([f3, b3]) for f3, b3 in zip(pre_fws3, reversed(pre_bws3))]

    pre_v_elemets_size = len(pre_b_tagtag[0].npvalue())
    pre_row_num = len(pre_b_tagtag)
    pre_vecs_concat = dy.concatenate([v for v in pre_b_tagtag])
    pre_mat = dy.reshape(pre_vecs_concat, (pre_v_elemets_size, pre_row_num))
    pre_final = dy.concatenate([dy.kmax_pooling(v, 1, 0) for v in pre_mat])

    hy_fws = fl1_init.transduce(hy_wembs)
    hy_bws = bl1_init.transduce(reversed(hy_wembs))

    hy_bi = [dy.concatenate([word, f, b]) for word, f, b in zip(hy_wembs, hy_fws, reversed(hy_bws))]
    hy_fws2 = fl2_init.transduce(hy_bi)
    hy_bws2 = bl2_init.transduce(reversed(hy_bi))

    hy_b_tag = [dy.concatenate([word, f1, b1, f2, b2]) for word, f1, b1, f2, b2 in zip(hy_wembs, hy_fws, reversed(hy_bws), hy_fws2, reversed(hy_bws2))]
    hy_fws3 = fl3_init.transduce(hy_b_tag)
    hy_bws3 = bl3_init.transduce(reversed(hy_b_tag))

    hy_b_tagtag = [dy.concatenate([f3, b3]) for f3, b3 in zip(hy_fws3, reversed(hy_bws3))]

    hy_v_elemets_size = len(hy_b_tagtag[0].npvalue())
    hy_row_num = len(hy_b_tagtag)
    hy_vecs_concat = dy.concatenate([v for v in hy_b_tagtag])
    hy_mat = dy.reshape(hy_vecs_concat, (hy_v_elemets_size, hy_row_num))
    hy_final = dy.concatenate([dy.kmax_pooling(v, 1, 0) for v in hy_mat])

    final = dy.concatenate([pre_final, hy_final, dy.abs(pre_final - hy_final), dy.cmult(pre_final, hy_final)])


    W1 = dy.parameter(holder.W1)
    b1 = dy.parameter(holder.b1)
    W2 = dy.parameter(holder.W2)
    b2 = dy.parameter(holder.b2)

    mid = dy.rectify(W1 * final + b1)

    return W2 * mid + b2


def predict_tags(pre_words, hy_words, holder):
    vec = build_graph(pre_words, hy_words, holder)
    vec = dy.softmax(vec)
    prob = vec.npvalue()
    tag = np.argmax(prob)
    return tag


def evaluate_set2(set, holder):
    good = 0.0
    bad = 0.0
    for i, (pre_sentence, hy_sentence, tag) in enumerate(set):
        dy.renew_cg()
        pre_words = [word for word in pre_sentence]
        hy_words = [word for word in hy_sentence]
        predicted_tag = predict_tags(pre_words, hy_words, holder)
        if tag == predicted_tag:
            good += 1
        else:
            bad += 1
    return good / (good + bad)



def calc_loss(pre_words, hy_words, tag,  holder):
    vec = build_graph(pre_words, hy_words, holder)
    loss = dy.pickneglogsoftmax(vec, tag)
    return loss

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)




def main():
    print 'reading and processing data'
    vocab, trainset = get_set('data/snli/processed_data/train.txt')
    _, devset = get_set('data/snli/processed_data/dev.txt')
    _, testset = get_set('data/snli/processed_data/test.txt')
    rare_words = get_rare_words(vocab, SNLI_VOCAB_SIZE)
    ensure_directory_exists('results')

    train_batches = split_into_batches(trainset, BATCH_SIZE)


    word2index = {}
    vocab_size = None
    word_embedding = None
    vectors = []
    if USE_PRETRINED:
        with open("data/glove/glove.840B.300d.txt") as f:
            f.readline()
            for i, line in enumerate(f):
                fields = line.strip().split(" ")
                word2index[fields[0]] = i
                vectors.append(list(map(float, fields[1:])))

        model = dy.Model()
        word_embedding = model.add_lookup_parameters((len(vectors), len(vectors[0])))
        word_embedding.init_from_array(np.array(vectors))
    else:
        vocab = set(vocab)
        for word in rare_words:
            vocab.remove(word)
        vocab.add(UNKNOWN_WORD)
        vocab = list(vocab)

        word2index = {c: i for i, c in enumerate(vocab)}
        vocab_size = len(vocab)



    if not USE_PRETRINED:
        model = dy.Model()
        word_embedding = model.add_lookup_parameters((vocab_size, WORD_EMBED_SIZE))
    trainer = dy.AdamTrainer(model)

    l1_hidden_dim = 128
    l2_hidden_dim = 256
    l3_hidden_dim = 512

    mlpd = 800


    fwdRNN_layer1 = dy.LSTMBuilder(layers=1, input_dim=WORD_EMBED_SIZE, hidden_dim=l1_hidden_dim, model=model)
    bwdRNN_layer1 = dy.LSTMBuilder(layers=1, input_dim=WORD_EMBED_SIZE, hidden_dim=l1_hidden_dim, model=model)

    fwdRNN_layer2 = dy.LSTMBuilder(layers=1, input_dim=WORD_EMBED_SIZE + l1_hidden_dim * 2, hidden_dim=l2_hidden_dim, model=model)
    bwdRNN_layer2 = dy.LSTMBuilder(layers=1, input_dim=WORD_EMBED_SIZE + l1_hidden_dim * 2, hidden_dim=l2_hidden_dim, model=model)

    fwdRNN_layer3 = dy.LSTMBuilder(layers=1, input_dim=WORD_EMBED_SIZE + (l1_hidden_dim + l2_hidden_dim) * 2, hidden_dim=l3_hidden_dim, model=model)
    bwdRNN_layer3 = dy.LSTMBuilder(layers=1, input_dim=WORD_EMBED_SIZE + (l1_hidden_dim + l2_hidden_dim) * 2, hidden_dim=l3_hidden_dim, model=model)

    W1 = model.add_parameters((mlpd, l3_hidden_dim * 2 * 4))
    b1 = model.add_parameters(mlpd)

    W2 = model.add_parameters((3, mlpd))
    b2 = model.add_parameters(3)

    holder = ComponentHolder(word2index, fwdRNN_layer1, bwdRNN_layer1, fwdRNN_layer2, bwdRNN_layer2, fwdRNN_layer3, bwdRNN_layer3, W1, b1, W2, b2, word_embedding)

    start_training_time = datetime.now()
    current_date = start_training_time.strftime("%d.%m.%Y")
    current_time = start_training_time.strftime("%H:%M:%S")
    print 'start time: {}'.format(start_training_time)

    for iter in xrange(EPOCHS):
        pretrain_time = datetime.now()
        random.shuffle(train_batches)
        for i, tuple in enumerate(train_batches, 1):
            dy.renew_cg()
            losses = []
            if i % 100 == 0:
                print 'working on epoch {}, batch {}/{}. time since start {}'.format(iter + 1, i, len(train_batches), datetime.now() - start_training_time)
            if i % EVALUATE_ITERATION == 0:
                print 'evaluating dev and test'
                print 'start evaluation time: {}'.format(datetime.now())
                dev_eval = evaluate_set2(devset, holder)
                test_eval = evaluate_set2(testset, holder)
                with open('results/' + method + '_' + dev_evaluation_file, 'a') as f:
                    f.write(' ' + str(dev_eval))
                with open('results/' + method + '_' + test_evaluation_file, 'a') as f:
                    f.write(' ' + str(test_eval))
                print 'finished evaluating'
                print 'finished evaluation time: {}'.format(datetime.now())
            for i, item in enumerate(tuple):
                pre_words = [word for word in item[0]]
                hy_words = [word for word in item[1]]
                tag = item[2]
                loss = calc_loss(pre_words, hy_words, tag,  holder)
                losses.append(loss)
            loss_exp = dy.esum(losses) / len(losses)
            loss_exp.backward()
            trainer.update()
        print 'epoch took: ' + str(datetime.now() - pretrain_time)

    total_training_time = datetime.now() - start_training_time
    print 'total training time was {}'.format(total_training_time)


if __name__ == '__main__':
    main()