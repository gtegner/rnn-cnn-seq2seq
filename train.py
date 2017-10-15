from __future__ import unicode_literals, print_function, division

__author__ = 'gustaftegner'

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


from CNNModels import ConvEncoder2,AttnDecoderRNN_CNN, CNNTest, DecoderRNN

from data_preprocessing import *

from utils import *

from sklearn.model_selection import train_test_split

from main import train_simple, trainIters_simple


def split_data(pairs):
    pairs = np.asarray(pairs)
    print(pairs.shape)


    N = len(pairs)
    indices = np.arange(N)
    np.random.shuffle(indices)
    train_p = 0.6
    val_p = 0.2
    test_p = 0.2


    train_ind = indices[0:int(train_p * N)]
    val_ind = indices[int(train_p*N)+1:int((train_p + val_p)*N)]
    test_ind = indices[int((train_p*val_p)*N)+1:]

    train = pairs[train_ind]
    val = pairs[val_ind]
    test = pairs[test_ind]

    return train, val, test




def create_model(input_lang,output_lang, model='simple'):

    hidden_size = 24
    embedding_dim = 32
    kernel_size = 3
    encoder_layers = 1

    if(model == 'fairseq'):
        encoder = ConvEncoder2(input_lang.n_words, hidden_size, embedding_dim, kernel_size, encoder_layers, MAX_LENGTH)
        decoder = AttnDecoderRNN_CNN(hidden_size, output_lang.n_words, n_layers=3, dropout_p=0.1)
    elif model=='simple':
        encoder = CNNTest(input_lang.n_words, hidden_size, embedding_dim, max_length=10)
        decoder = DecoderRNN(output_lang.n_words, hidden_size, n_layers=1)

    return encoder, decoder


def get_loss(input_variable, target_variable, encoder, decoder,
                    encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]


    loss = 0


    encoder_outputs, encoder_hidden = encoder(input_variable)

    decoder_hidden = encoder_hidden[:,-1,:].unsqueeze(1)

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input



    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs, encoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        #print("DECODER OUTPUT: ", decoder_output)
        #print("TARGET VARIABLE: ", target_variable[di])
        loss += criterion(decoder_output, target_variable[di])
        if ni == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def train(data,input_lang,output_lang,encoder,decoder,epochs,print_every=1000,plot_every=100,
               learning_rate = 0.01):

    pairs = data
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)
    training_pairs = [variablesFromPair(random.choice(pairs), input_lang, output_lang) for i in range(epochs)]

    criterion = nn.NLLLoss()

    for iter in range(1,epochs+1):
        training_pair = training_pairs[iter-1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = get_loss(input_variable, target_variable, encoder, decoder,
                    encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss



        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / epochs),
                                         iter, iter / epochs * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


def main():

    input_lang,output_lang, pairs = prepareData('eng','fra',True)
    #train_data,val_data, test_data = split_data(pairs)

    train_data = pairs
    models = ['simple', 'fairseq']

    model = models[1]
    encoder, decoder = create_model(input_lang,output_lang,model)

    epochs = 1000
    print_every = 100
    if(model == 'simple'):
        trainIters_simple(train_data,input_lang, output_lang, encoder,decoder,epochs,print_every)
    else:
        train(train_data,input_lang, output_lang, encoder,decoder,epochs,print_every)


main()