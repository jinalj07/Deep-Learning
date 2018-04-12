from problem1 import SoftmaxRegression as sr
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
#-------------------------------------------------------------------------
'''
    Problem 3: Recurrent Neural Network 
    In this problem, we will implement the recurrent neural network for sequence classification problems.
    We will use cross entropy as the loss function and stochastic gradient descent to train the model parameters.
    Did NOT use torch.nn.RNN, implemented my own version of RNN using only basic tensor operations.
'''


#--------------------------
def tanh(z):
    """ Computing the hyperbolic tangent of the elements
        math: f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
        Input:
            z:  a batch of training instances, a float torch Variable of shape (n by p). n is the number instances in a batch. p is the number of features.
        Output:
            a: the nonlinear activation tensor, a float torch Variable of shape (n by n_filters by h by w )
        Note: did NOT use th.nn.Tanh, th.tanh, th.nn.functional.tanh, implemented my own version using basic tensor operations, such as exp(), div(), etc.
    """

    z1 = th.clamp(z,min = -10, max = 10)
       
    e1 = th.exp(z1)-th.exp(-z1)
    e2 = th.exp(z1)+th.exp(-z1)
    a = th.div(e1,e2)

    return a


#-------------------------------------------------------
class RNN(sr):
    '''RNN is a recurrent neural network with hyperbolic tangent function as the activation function.
       After the recurrent layer, we apply a fully connected layer from hidden neurons to produce the output.
    '''
    # ----------------------------------------------
    def __init__(self, p, h=10, c=10):
        ''' Initializing the model. Creating parameters of recurrent neural network. 
            Inputs:
                p: the number of input features, an integer scalar
                h: the number of memory (hidden) neurons, an integer scalar
                c: the number of output classes, an integer scalar
            Outputs:
                self.U: the weight matrix connecting the input features to the hidden units, a torch Variable of shape h by p, initialized as all-zeros. 
                self.V: the weight matrix connecting the hidden units to hidden units, a torch vector Variable of shape h by h, initialized as all-zeros.
                self.b_h: the bias vector of the hidden units, a torch vector Variable of length h, initialized as all-ones, to avoid vanishing gradient.
                self.W: the weight matrix parameter in fully connected layer from hidden units to the output prediction, a torch Variable of shape (p, c), initialized as all-zeros. 
                        Hint: RNN is a subclass of SoftmaxRegression, which already has a W parameter and b. 
                self.b: the bias vector parameter of the outputs, a torch Variable of shape (c), initialized as all-zeros
                self.loss_fn: the loss function object for softmax regression. 
            You could solve this problem using 4 lines of code.
        '''

        self.U = Variable(th.zeros(p,h), requires_grad=True)
        self.V = Variable(th.zeros(h,h), requires_grad=True)
        self.b_h = Variable(th.ones(h), requires_grad=True)
        super(RNN,self).__init__(h, c)



    # ----------------------------------------------
    def forward(self, x, H):
        '''
           Given a batch of training instances (with one time step), computing the linear logits z in the outputs.
            Input:
                x:  a batch of training instance, a float torch Tensor of shape n by p. Here n is the batch size. p is the number of features. 
                H:  the hidden state of the RNN model, a float torch Tensor of shape  n by h. Here h is the number of hidden units. 
            Output:
                z: the logit values of the batch of training instances after the output layer, a float matrix of shape n by c. Here c is the number of classes
                H_new: the new hidden state of the RNN model, a float torch Tensor of shape n by h.
        '''

        x1 = th.mm(x,self.U)
        h1 = th.mm(H,self.V)+self.b_h
        x2 = x1 + h1
        z = th.mm(tanh(x2),self.W)+self.b
        H_new = tanh(x2)

        return z, H_new


    # ----------------------------------------------
    def train(self, loader, n_steps=10,alpha=0.01):
        """training the model 
              Input:
                loader: dataset loader, which loads one batch of dataset at a time.
                        x: a batch of training instance, a float torch Tensor of shape n by t by p. Here n is the batch size. p is the number of features. t is the number of time steps.
                        y: a batch of training labels, a torch LongTensor of shape n by t. 
                n_steps: the number of batches of data to train, an integer scalar. Note: the n_steps is the number of training steps, not the number of time steps (t).
                alpha: the learning rate for SGD(stochastic gradient descent), a float scalar
                Note: the loss of a sequence is computed as the sum of the losses in all time steps of the sequence.
        """
        # create a SGD optimizer
        optimizer = SGD([self.U,self.V,self.b_h,self.W,self.b], lr=alpha)
        count = 0
        while True:
            # use loader to load one batch of training data
            for x,y in loader:
                # convert data tensors into Variables
                x = Variable(x)
                y = Variable(y)
                n,t,p = x.size()
                h,_ = self.V.size()
                # initialize hidden state as all zeros
                H = Variable(th.zeros(n,h))

                L = 0
                # go through each time step
                for i in range(t):
                    # forward pass
                    z, H = self.forward(x[:,i,:],H)
                    # compute loss 
                    L += self.compute_L(z,y[:,i])
                # backward pass: compute gradients
                L.backward()
                # update model parameters
                optimizer.step()
                # reset the gradients to zero
                optimizer.zero_grad()

                count+=1
                if count >=n_steps:
                    return
