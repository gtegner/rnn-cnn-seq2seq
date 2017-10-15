
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

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]



    loss = 0

    encoder_output = encoder(input_variable)


    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_output


    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input,decoder_hidden)
        topv, topi = decoder_output.data.topk(1)

        ni = topi[0][0]
        a = torch.LongTensor([[ni]])

        decoder_input = Variable(torch.LongTensor([[ni]]))

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

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]


    loss = 0



    encoder_output = encoder(input_variable)

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))


    decoder_hidden = encoder_output

    outputs = [SOS_token]

    for di in range(target_length):
        do = decoder(decoder_input,decoder_hidden)
        #print("Decoder output ", do)

        topv, topi = do.data.topk(1)

        ni = topi[0][0]
        a = torch.LongTensor([[ni]])


        outputs.append(ni)
        decoder_input = Variable(torch.LongTensor(outputs)).view(-1,1)

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


    loss = 0

    encoder_outputs, encoder_hidden = encoder(input_variable)

    decoder_hidden = encoder_hidden[:,-1,:].unsqueeze(1)


    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input


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





