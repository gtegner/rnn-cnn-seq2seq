
# coding: utf-8

# In[2]:

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from utils import *

from CNNModels import *

from data_preprocessing import *

use_cuda = torch.cuda.is_available()



teacher_forcing_ratio = 0.0

# In[79]:




def train_simple(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    #encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    #encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    #encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    encoder_output = encoder(input_variable)


    #for ei in range(input_length):
    #    encoder_output, encoder_hidden = encoder(
    #        input_variable[ei], encoder_hidden)
    #    encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_output


    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input,decoder_hidden)
        topv, topi = decoder_output.data.topk(1)

        ni = topi[0][0]
        a = torch.LongTensor([[ni]])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        #decoder_input = Variable(torch.LongTensor[[ni]])
        #print("TARGET VARIABLE ", target_variable[di])
        #print("DECODER OUTPUT", decoder_output)
        #print("TARGET VARIABLE", target_variable[di])
        loss += criterion(decoder_output, target_variable[di])
        if(loss.data[0] / target_length > 100):
            print("LOSS is big")

        if ni == EOS_token:
            break

    loss.backward()


    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def train_conv(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    #encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    #encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    #encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    #print(target_variable)

    encoder_output = encoder(input_variable)

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    #decoder_input = target_variable

    decoder_hidden = encoder_output

    #print("encoder output ", decoder_hidden)
    outputs = [SOS_token]

    for di in range(target_length):
        do = decoder(decoder_input,decoder_hidden)
        #print("Decoder output ", do)

        topv, topi = do.data.topk(1)

        ni = topi[0][0]
        #ni = topi[0]
        a = torch.LongTensor([[ni]])

        #print(a)
        #print(decoder_output)

        outputs.append(ni)
        decoder_input = Variable(torch.LongTensor(outputs)).view(-1,1)
        #print("decoder input ", decoder_input)

        do = do.view(1,-1)

        loss += criterion(do, target_variable[di])
        if(loss.data[0] / target_length > 100):
            print("LOSS is big")

        if ni == EOS_token:
            break

    loss.backward()


    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def trainIters_conv(data,encoder,decoder,n_iters,print_every=1000,plot_every=100,
               learning_rate = 0.01):

    pairs = data
    print("Training started...")
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.Adagrad(encoder.parameters())
    decoder_optimizer = optim.Adagrad(decoder.parameters())
    training_pairs = [variablesFromPair(random.choice(pairs)) for i in range(n_iters)]

    criterion = nn.NLLLoss()

    for iter in range(1,n_iters+1):
        training_pair = training_pairs[iter-1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train_conv(input_variable, target_variable, encoder, decoder,
                    encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss


        if(print_loss_total / print_every > 100):
            print("Loss is very big :D")

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

def trainIters_simple(data,input_lang,output_lang,encoder,decoder,n_iters,print_every=1000,plot_every=100,
               learning_rate = 0.01):

    pairs = data
    print("Training started...")
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.Adagrad(encoder.parameters())
    decoder_optimizer = optim.Adagrad(decoder.parameters())
    training_pairs = [variablesFromPair(random.choice(pairs),input_lang,output_lang) for i in range(n_iters)]

    criterion = nn.NLLLoss()

    for iter in range(1,n_iters+1):
        training_pair = training_pairs[iter-1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train_simple(input_variable, target_variable, encoder, decoder,
                    encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss


        if(print_loss_total / print_every > 100):
            print("Loss is very big :D")

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0





def train_attn(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    #encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    #encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    #for ei in range(input_length):
    #    encoder_output, encoder_hidden = encoder(
    #        input_variable[ei], encoder_hidden)
    #    encoder_outputs[ei] = encoder_output[0][0]

    encoder_outputs, encoder_hidden = encoder(input_variable)

    decoder_hidden = encoder_hidden[:,-1,:].unsqueeze(1)


    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    #decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, encoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
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


def trainIters_attn(data,encoder,decoder,n_iters,print_every=1000,plot_every=100,
               learning_rate = 0.01):

    pairs = data
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)
    training_pairs = [variablesFromPair(random.choice(pairs)) for i in range(n_iters)]

    criterion = nn.NLLLoss()

    for iter in range(1,n_iters+1):
        training_pair = training_pairs[iter-1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train_attn(input_variable, target_variable, encoder, decoder,
                    encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss



        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0



def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]

    #encoder_hidden = encoder.initHidden()

    #encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    #encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs


    encoder_output, encoder_hidden = encoder(input_variable)

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden[:,-1,:].unsqueeze(1)
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attentions = decoder(decoder_input,decoder_hidden, encoder_output, encoder_hidden)

        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        if ni == PAD_token:
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words


def evaluateRandomly(encoder, decoder,pairs, n=20):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import numpy as np

def main():
    input_lang,output_lang, pairs = prepareData('eng','fra',True)


    hidden_size = 10
    n_iters=2
    embedding_dim = 12
    kernel_size = 2

    cnn_encoder = simple_CNN_encoder(input_lang.n_words, hidden_size, embedding_dim, kernel_size)

    #cnn_encoder = ConvEncoder(input_lang.n_words, hidden_size, kernel_size, n_layers=1)
    decoder1 = DecoderRNN(output_lang.n_words,hidden_size)


    print("Training...")

    trainIters_simple(pairs,input_lang,output_lang,cnn_encoder, decoder1, n_iters, print_every=1)

    evaluateRandomly(cnn_encoder, decoder1)


def main2():
    input_lang,output_lang, pairs = prepareData('eng','fra',True)


    hidden_size = 32
    n_iters=1000
    embedding_dim = 12
    kernel_size = 3

    cnn_encoder = simple_CNN_encoder(input_lang.n_words, hidden_size, embedding_dim, kernel_size)
    decoder1 = DecoderRNN(output_lang.n_words,hidden_size)

    conv_encoder = ConvEncoder(input_lang.n_words, hidden_size, kernel_size, n_layers=1)
    conv_decoder = ConvDecoder(output_lang.n_words, hidden_size, kernel_size, n_layers=1)

    print("Training...")

    trainIters_conv(pairs,conv_encoder, conv_decoder, n_iters, print_every=100)

    print("Done")
    #evaluateRandomly(conv_encoder, conv_decoder)


def test():
    input_lang,output_lang, pairs = prepareData('eng','fra',True)

    pair1 = random.choice(pairs)
    print_pair(pair1)
    inp, target = variablesFromPair(pair1)


    kernel_size = 3
    hidden_size = 16
    embedding_dim = 8

    print("output_lang words ", output_lang.n_words)
    encoder = ConvEncoder(input_lang.n_words, hidden_size, kernel_size,1)
    decoder = ConvDecoder(output_lang.n_words, hidden_size, kernel_size, n_layers=1)

    #encoder = CNNTest(input_lang.n_words, hidden_size, embedding_dim, 2)
    encoder_output = encoder(inp)
    print(encoder_output.size())


    #decoder_input = Variable(torch.LongTensor([[SOS_token]]))

    #decoder_hidden = encoder_output

    #decoder_output = decoder(decoder_input,decoder_hidden)

    #print(decoder_output)

    #a,b = decoder_output.data.topk(1)

    #print("a")
    #print(a)
    #print("b")
    #print(b)



def main3():
    input_lang,output_lang, pairs = prepareData('eng','fra',True)

    pair1 = random.choice(pairs)
    print_pair(pair1)
    inp, target = variablesFromPair(pair1)

    hidden_size = 8
    embedding_dim = 8
    kernel_size = 3
    n_iters = 10000

    encoder = ConvEncoder(input_lang.n_words, hidden_size, embedding_dim, kernel_size)

    decoder = AttnDecoderRNN_CNN(hidden_size, output_lang.n_words, n_layers=1,
                               dropout_p=0.1)

    #print(output)

    trainIters_attn(pairs,encoder,decoder,n_iters, print_every= 100)
    evaluateRandomly(encoder,decoder, pairs)


def mainConv():
    input_lang,output_lang, pairs = prepareData('eng','sv',True)

    pair1 = random.choice(pairs)
    print_pair(pair1)
    inp, target = variablesFromPair(pair1)

    hidden_size = 12
    embedding_dim = 16
    kernel_size = 3
    n_iters = 100000

    encoder = GatedConvLayer(input_lang.n_words, hidden_size, embedding_dim, kernel_size, n_layers=1)

    out = encoder(inp)

    #print(out)

#main3()






