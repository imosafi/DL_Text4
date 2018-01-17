import dynet as dy
from datetime import datetime
import random
import os
import sys
from enum import Enum
from collections import Counter
import numpy as np
import gc

# WORD_EMBED_SIZE = 300
SNLI_VOCAB_SIZE = 40000 # 42378 - real size
BATCH_SIZE = 500 #250
WORD_EMBED_SIZE = 120 # 300 IN PYTOCH MODEL
EPOCHS = 2
EVALUATE_ITERATION = 1#500

CHATS_TO_SKIP = ',?'
SEPERATION_STR = '$$$'
UNKNOWN_WORD = 'UnKnOwN_wOrD'

labelDict = {'neutral': 0, 'entailment': 1, 'contradiction': 2, '-': 3}


class TrainingOn(Enum):
    snli = 1
    multinli = 2
    both = 3


# args[1] => use pretrained
def make_sure_labels_are_0_2(trainset):
    return [t for t in trainset if t[2] != 3]


def get_rare_words(words, vocab_size):
    counter = Counter(words)
    most_common_keys = [x[0] for x in counter.most_common(vocab_size)]
    keys_to_remove = [x for x in counter.keys() if x not in most_common_keys]
    return keys_to_remove


def split_into_batches(set, size):
    return [set[i:i + size] for i in range(0, len(set), size)]


# naive implementation don't remove , ? etc.
def get_train_set(file_name):
    trainset = []
    vocab = []
    for line in file(file_name):
        line = line.strip(' \t\n')
        parts = line.split(SEPERATION_STR)
        if not len(parts) == 3:
            raise Exception("more than 3 parts")
        # if len(parts) == 2:
        #     parts.append(parts[1][- 1])
        #     # parts[2] = parts[1][- 1]
        #     parts[1] = parts[1][::len(parts[1]) - 1]
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
    trainset = make_sure_labels_are_0_2(trainset)
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
    # dy.renew_cg()
    fl1_init = holder.fwdRNN_layer1.initial_state()
    bl1_init = holder.bwdRNN_layer1.initial_state()
    fl2_init = holder.fwdRNN_layer2.initial_state()
    bl2_init = holder.bwdRNN_layer2.initial_state()
    fl3_init = holder.fwdRNN_layer3.initial_state()
    bl3_init = holder.bwdRNN_layer3.initial_state()

    pre_wembs = [get_word_rep(w, holder) for w in pre_words]
    hy_wembs = [get_word_rep(w, holder) for w in hy_words]

    # if is_training:
    #     wembs = [dy.dropout(w, 0.1) for w in wembs]
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
    # needs to be fixed (another layer + dropout)
    mid = dy.rectify(W1 * final + b1)
    # if is_training:
    #     b_tag = [dy.dropout(b, 0.1) for b in b_tag]
    return W2 * mid + b2


def predict_tags(pre_words, hy_words, holder):
    vec = build_graph(pre_words, hy_words, holder)
    vec = dy.softmax(vec)
    prob = vec.npvalue()
    tag = np.argmax(prob) # make sure it returns the index
    # tags.append(holder.index2tag[tag])
    return tag

def evaluate_testset(testset, holder):
    good = 0.0
    bad = 0.0
    for i, (pre_sentence, hy_sentence, tag) in enumerate(testset):
        # gc.collect()
        dy.renew_cg()
        print 'evaluate item {}'.format(i)
        if i % 100 == 1:
            print 'good / (good + bad) = {}'.format(good / (good + bad))
        pre_words = [word for word in pre_sentence]
        hy_words = [word for word in hy_sentence]

        predicted_tag = predict_tags(pre_words, hy_words, holder)
        if tag == predicted_tag:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def evaluate_set(dev_batches, holder):
    good = 0.0
    bad = 0.0
    for i, item in enumerate(dev_batches):
        gc.collect()
        print 'evaluate batch {}'.format(i)
        for pre_sentence, hy_sentence, tag in item:
            pre_words = [word for word in pre_sentence]
            hy_words = [word for word in hy_sentence]

            predicted_tag = predict_tags(pre_words, hy_words, holder)
            if tag == predicted_tag:
                good += 1
            else:
                bad += 1
    return good / (good + bad)

# maybe write calc_batch_loss and calculate the loss for the whole batch
def calc_loss(pre_words, hy_words, tag,  holder):
    vec = build_graph(pre_words, hy_words, holder)
    # losses = []
    # tid = holder.tag2index[t]
    loss = dy.pickneglogsoftmax(vec, tag)
    return loss
    # losses.append(loss)
    # return dy.esum(losses)

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def save_results_and_model(evaluation_results, current_date, current_time, total_training_time, model):
    path = 'results/{}_{}'.format(current_date, current_time)
    ensure_directory_exists(path)
    with open(path + '/results.txt', 'w') as f:
        f.write('total training time: {}\n'.format(total_training_time))
        f.write('num of epochs: {}\n'.format(EPOCHS))
        f.write('evaluated every: {}\n'.format(EVALUATE_ITERATION))
    with open(path + '/' + 'evaluation.txt', 'w') as f:
        f.write(''.join([str(x) + ' ' for x in evaluation_results]))
    model.save(path + '/' + 'model1')


def main():
    # if sys.argv[1] == 1:
    #     return None
    # else:
    vocab, trainset = get_train_set('data/snli/sequence/train.txt')
    # _, devset = get_train_set('data/snli/sequence/dev.txt')
    _, testset = get_train_set('data/snli/sequence/test.txt')
    rare_words = get_rare_words(vocab, SNLI_VOCAB_SIZE)

    train_batches = split_into_batches(trainset, BATCH_SIZE)
    # dev_batches = split_into_batches(devset, BATCH_SIZE)
    test_bathces = split_into_batches(testset, BATCH_SIZE)

    vocab = set(vocab)
    for word in rare_words:
        vocab.remove(word)
    vocab.add(UNKNOWN_WORD)
    vocab = list(vocab)

    word2index = {c: i for i, c in enumerate(vocab)}
    vocab_size = len(vocab)

    # create label2index - not needed since the labels are exactly the numbers

    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    l1_hidden_dim = 200
    l2_hidden_dim = 500
    l3_hidden_dim = 800

    mlpd = 1000

    word_embedding = model.add_lookup_parameters((vocab_size, WORD_EMBED_SIZE))

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

    # evaluation_results = []
    for iter in xrange(EPOCHS):
        pretrain_time = datetime.now()
        random.shuffle(train_batches)
        for i, tuple in enumerate(train_batches, 1):
            print 'working on batch {}. time since start {}'.format(i, datetime.now() - start_training_time)
            dy.renew_cg()
            losses = []
            # if i % EVALUATE_ITERATION == 0:
            #     eval = evaluate_set(dev_batches, holder)
            #     evaluation_results.append(eval)
            #     print 'epoch {}, batch {}, validation evaluation {}'.format(iter + 1, i, eval)
            for i, item in enumerate(tuple):
                pre_words = [word for word in item[0]]
                hy_words = [word for word in item[1]]
                tag = item[2]
                loss = calc_loss(pre_words, hy_words, tag,  holder)
                losses.append(loss)
            loss_exp = dy.esum(losses) / len(losses)  # check if should be converted to float
            loss_exp.backward()
            trainer.update()
        print 'epoch took: ' + str(datetime.now() - pretrain_time)

    total_training_time = datetime.now() - start_training_time
    print 'total training time was {}'.format(total_training_time)

    print 'test result is {}'.format(evaluate_testset(testset, holder))

    # save_results_and_model(evaluation_results, current_date, current_time, total_training_time, model)


if __name__ == '__main__':
    main()