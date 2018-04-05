from problem1 import SoftmaxRegression as sr
from problem3 import tanh
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
#-------------------------------------------------------------------------
'''
    Problem 4: LSTM 
    In this problem, we will implement the LSTM for sequence classification problems.
    We will use cross entropy as the loss function and stochastic gradient descent to train the model parameters.
    did NOT use torch.nn.LSTM, implemented my own version of LSTM using only basic tensor operations.
'''

#-------------------------------------------------------
class LSTM(sr):
    '''LSTM is a recurrent neural network with Long-Short Term Memory.  '''
 
    # ----------------------------------------------
    def __init__(self, p, h=10, c=10):
        ''' Initializing the model. Creating parameters of recurrent neural network. 
            Inputs:
                p: the number of input features, an integer scalar
                h: the number of memory (hidden) neurons, an integer scalar
                c: the number of output classes, an integer scalar
            Outputs:
                self.W_i: the weight matrix of the input gate, a torch Variable of shape (p+h) by h, initialized as all-zeros. 
                self.b_i: the biases of the input gate, a torch Variable of length h, initialized as all-zeros. 
                self.W_o: the weight matrix of the output gate, a torch Variable of shape (p+h) by h, initialized as all-zeros. 
                self.b_o: the biases of the output gate, a torch Variable of length h, initialized as all-zeros. 
                self.W_c: the weight matrix of generating candidate cell states, a torch Variable of shape (p+h) by h, initialized as all-zeros. 
                self.b_c: the biases of generating candidate cell states, a torch Variable of length h, initialized as all-zeros. 
                self.W_f: the weight matrix of the forget gate, a torch Variable of shape (p+h) by h, initialized as all-zeros. 
                self.b_f: the biases of the forget gate, a torch Variable of length h, initialized as all-zeros. 
                self.W: the weight matrix parameter in fully connected layer from hidden unit to the output, a torch Variable of shape (h, c), initialized as all-zeros. 
                        Hint: LSTM is a subclass of SoftmaxRegression, which already has a W parameter and b. 
                self.b: the bias vector parameter of the outputs, a torch Variable of shape (c), initialized as all-zeros
                self.loss_fn: the loss function object for softmax regression. 
        '''
        self.W_i = Variable(th.zeros((p+h),h), requires_grad=True)
        self.b_i = Variable(th.zeros(h), requires_grad=True)
        self.W_o = Variable(th.zeros((p+h),h), requires_grad=True)
        self.b_o = Variable(th.zeros(h), requires_grad=True)
        self.W_c = Variable(th.zeros((p+h),h), requires_grad=True)
        self.b_c = Variable(th.zeros(h), requires_grad=True)
        self.W_f = Variable(th.zeros((p+h),h), requires_grad=True)
        self.b_f = Variable(th.zeros(h), requires_grad=True)
        super(LSTM,self).__init__(h, c)


    # ----------------------------------------------
    def gates(self, x, H):
        '''
           Given a batch of training instances (with one time step), computing the gating functions: forget gate f, input gate i, output gate o; and candidate cells state.
           Note: W_f is matrix consisting weights for both inputs and hidden states. Here we assume the first p rows of W_f coresponds to input weights, last h rows of W_f coresponds to the weights of hidden states. 
            Input:
                x:  a batch of training instance, a float torch Tensor of shape n by p. Here n is the batch size. p is the number of features. 
                H:  the hidden state of the LSTM, a float torch Tensor of shape  n by h. Here h is the number of hidden units. 
            Output:
                f: the forget gate values of the batch of training instances, a float matrix of shape n by h. Here h is the number of hidden units.
                i: the input gate values of the batch of training instances, a float matrix of shape n by h. Here h is the number of hidden units.
                o: the output gate values of the batch of training instances, a float matrix of shape n by h. Here h is the number of hidden units.
                C_c: the candidate cell state values of the batch of training instances, a float matrix of shape n by h. Here h is the number of hidden units.
            Hint: you could solve this problem using 4-5 lines of code.
        '''
        f = th.sigmoid(th.mm((th.cat([x,H],1)),self.W_f)+self.b_f)
        i = th.sigmoid(th.mm(th.cat([x,H],1),self.W_i)+self.b_i)
        o = th.sigmoid(th.mm(th.cat([x,H],1),self.W_o)+self.b_o)
        C_c = tanh(th.mm(th.cat([x,H],1),self.W_c)+self.b_c)

        return f, i, o, C_c


    # ----------------------------------------------
    @staticmethod
    def update_cell(C, C_c, f, i):
        '''
            Updating the cell state.
            Input:
                C:  the current cell state of the LSTM, a float torch Tensor of shape  n by h. Here h is the number of cell units. 
                C_c:  the candidate cell state of the LSTM, a float torch Tensor of shape  n by h. Here h is the number of cell units. 
                f: the forget gate values of the batch of training instances, a float matrix of shape n by h. Here h is the number of hidden units.
                i: the input gate values of the batch of training instances, a float matrix of shape n by h. Here h is the number of hidden units.
            Output:
                C_new: the updated cell state values of the batch of training instances, a float matrix of shape n by h. Here h is the number of hidden units.
        '''

        C_new = (th.mul(C,f))+(th.mul(C_c,i))

        return C_new 


    # ----------------------------------------------
    @staticmethod
    def output_hidden_state(C, o):
        '''
            Outputing the hidden state.
            Input:
                C:  the current cell state of the LSTM, a float torch Tensor of shape  n by h. Here h is the number of cell units. 
                o: the output gate values of the batch of training instances, a float matrix of shape n by h. Here h is the number of hidden units.
            Output:
                H: the output hidden state values of the batch of training instances, a float matrix of shape n by h. Here h is the number of hidden units.
        '''

        c1 = tanh(C)
        H = th.mul(c1,o)

        return H



    # ----------------------------------------------
    def forward(self, x, H,C):
        '''
           Given a batch of training instances (with one time step), computing the linear logits z in the outputs.
            Input:
                x:  a batch of training instance, a float torch Tensor of shape n by p. Here n is the batch size. p is the number of features. 
                H:  the hidden state of the LSTM model, a float torch Tensor of shape  n by h. Here h is the number of hidden units. 
                C:  the cell state of the LSTM model, a float torch Tensor of shape  n by h. 
            Output:
                z: the logit values of the batch of training instances after the output layer, a float matrix of shape n by c. Here c is the number of classes
                H_new: the new hidden state of the LSTM model, a float torch Tensor of shape n by h.
                C_new: the new cell state of the LSTM model, a float torch Tensor of shape n by h.
        '''

        f, i, o, C_c = self.gates(x, H)
        C_new = self.update_cell(C, C_c, f, i)
        H_new = self.output_hidden_state(C_new, o)
        z = H_new.mm(self.W)+self.b

        return z, H_new, C_new

    # ----------------------------------------------
    def train(self, loader, n_steps=10,alpha=0.01):
        """train the model 
              Input:
                loader: dataset loader, which loads one batch of dataset at a time.
                        x: a batch of training instance, a float torch Tensor of shape n by t by p. Here n is the batch size. p is the number of features. t is the number of time steps.
                        y: a batch of training labels, a torch LongTensor of shape n by t. 
                n_steps: the number of batches of data to train, an integer scalar. Note: the n_steps is the number of training steps, not the number of time steps (t).
                alpha: the learning rate for SGD(stochastic gradient descent), a float scalar
                Note: the loss of a sequence is computed as the sum of the losses at different time steps of the sequence.
        """
        # create a SGD optimizer
        optimizer = SGD([self.W_f,
                         self.W_i,
                         self.W_o,
                         self.W_c,
                         self.b_f,
                         self.b_i,
                         self.b_o,
                         self.b_c,
                         self.W,
                         self.b], lr=alpha)
        count = 0
        while True:
            # use loader to load one batch of training data
            for x,y in loader:
                # convert data tensors into Variables
                x = Variable(x)
                y = Variable(y)
                n,t,p = x.size()
                _,h = self.W_f.size()
                # initialize hidden state as all zeros
                H = Variable(th.zeros(n,h))
                C = Variable(th.zeros(n,h))

                L = 0
                # go through each time step
                for i in range(t):
                    # forward pass
                    z, H, C = self.forward(x[:,i,:], H, C)
                    # compute loss 
                    L += self.compute_L(z,y[:,i])
                # backward pass: compute gradients
                self.backward(L)
                # update model parameters
                optimizer.step()
                # reset the gradients to zero
                optimizer.zero_grad()

                count+=1
                if count >=n_steps:
                    return

