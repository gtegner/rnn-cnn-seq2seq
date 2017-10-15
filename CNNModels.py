__author__ = 'gustaftegner'
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 10

from data_preprocessing import PAD_token




class GatedConvLayer(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, kernel_size, n_layers, max_length=10):
        super(GatedConvLayer, self).__init__()

        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.layers = n_layers

        self.embed = nn.Embedding(input_size, embedding_dim, padding_idx=PAD_token)

        self.conv_w = nn.Conv2d(1, hidden_size, kernel_size=(kernel_size, embedding_dim))
        self.conv_v = nn.Conv2d(1, hidden_size, kernel_size=(kernel_size, embedding_dim))






    def create_embed(self, x):
        x = self.embed(x)
        x = x.transpose(0,1) #BatchxLengthxEmbedding
        x = x.transpose(1,2) #BatchxEmbeddingxLength
        mask_layer = torch.ones(x.size())
        mask_layer[:,:,:self.kernel_size,:] = 0
        mask_layer = Variable(mask_layer)

        x = x*mask_layer

        return x


    def forward(self, x):

        x = self.create_embed(x)


        for i in range(self.layers):
            x1 = self.conv_w(x)
            x2 = self.conv_v(x)
            out = x1 * F.sigmoid(x2)
            print(out.size())

        return out





class simple_CNN_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, max_length=10):
        super(simple_CNN_encoder, self).__init__()

        self.num_filters = 16
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.kernel_size = 2
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, self.embedding_dim, padding_idx=PAD_token)

        self.conv2 = nn.Conv2d(1,self.num_filters,(self.embedding_dim,self.kernel_size),bias=True)
        self.conv3 = nn.Conv2d(1,self.num_filters, (self.num_filters, self.kernel_size))
        self.conv_layers = [self.conv2, self.conv3]

        self.n_layers = len(self.conv_layers)

        self.max_pool1d = nn.MaxPool1d(2,stride=2)

        self.padding = nn.ZeroPad2d((1,1,1,1))

        self.linear1 = nn.Linear(int(self.num_filters*int(self.max_length/(2**self.n_layers))),hidden_size,bias=True)


    def oneLayer(self, input, convLayer):

        out = convLayer(input)
        out = F.relu(out)
        out = out.transpose(1,2)
        out = out.view(1,self.num_filters,-1)
        out = self.max_pool1d(out)
        out = out.view(1,1,self.num_filters,-1)

        return out

    def forward(self,input):
        input_length = input.size()[0]
        #print("Input shape ", input.size())
        #print("embedding size, ", self.embedding(input).size())
        embedded = self.embedding(input).view(1,1,self.max_length,-1)
        #print("After embed ", embedded.size())

        embedded = F.pad(embedded,(0,0,0,1))
        embedded = embedded.transpose(2,3)
        out = embedded

        for conv in self.conv_layers:
            out = self.oneLayer(out,conv)

        out = out.view(1,int(self.num_filters * int(self.max_length/2**self.n_layers)))
        out = self.linear1(out)
        out = out.view(1,1,-1)

        return out








class EncoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size,n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded
        for i in range(self.n_layers):
            output,hidden = self.gru(output,hidden)


        return output,hidden

    def initHidden(self):
        result = Variable(torch.zeros(1,1,self.hidden_size))

        return result


# In[56]:

class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers=1):
        super(DecoderRNN,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size,hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input,hidden):
        output = self.embedding(input).view(1,1,-1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output,hidden = self.gru(output,hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1,1,self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


# In[86]:

class AttnDecoderRNN_CNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN_CNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.lin1 = nn.Linear(self.hidden_size,self.hidden_size)





    def forward(self, input, hidden, encoder_outputs, encoder_contexts):
        embedded = self.embedding(input).view(1,1,-1)
        embedded = self.dropout(embedded)


        d = self.lin1(hidden)[0] + embedded[0] #1xHidden

        attn_vector = torch.bmm(encoder_outputs.unsqueeze(0), d.transpose(0,1).unsqueeze(0))

        attn_weights = F.softmax(attn_vector)

        contexts = torch.bmm(attn_weights.transpose(1,2), encoder_contexts)

        output = torch.cat((embedded[0],contexts[0]),1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output,hidden)

        output = F.log_softmax(self.out(output[0]))

        return output,hidden,attn_weights




    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class ConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, kernel_size,n_layers,max_length=10):
        super(ConvEncoder, self).__init__()

        self.num_filters = hidden_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim

        paddin = kernel_size-1

        self.num_layers = n_layers

        self.embed = nn.Embedding(input_size, self.embedding_dim, padding_idx=PAD_token)
        self.convA = nn.Conv1d(hidden_size, self.num_filters, kernel_size,padding=paddin)
        self.convB = nn.Conv1d(hidden_size, self.num_filters, kernel_size,padding = paddin)

        self.linear = nn.Linear(embedding_dim,hidden_size)

    def forward(self, x):
        x = self.embed(x)



        x = x.transpose(0,1) #BatchxLengthxEmbedding
        x = self.linear(x) #BatchxLengthxHidden

        x = x.transpose(1,2) #BatchxHiddenxLength


        cnn_a_input = x
        cnn_b_input = x

        for i in range(self.num_layers):
            cnn_a_input = self.convA(cnn_a_input)[:,:,:cnn_a_input.size(2)]
            cnn_a_input = F.tanh(cnn_a_input)

            cnn_b_input = self.convB(cnn_b_input)[:,:,:cnn_b_input.size(2)]
            cnn_b_input = F.tanh(cnn_b_input)


        A = cnn_a_input.squeeze(0).transpose(0,1)
        B = cnn_b_input.transpose(1,2)

        return (A,B)