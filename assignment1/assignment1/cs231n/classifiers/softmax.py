from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train=X.shape[0]
    num_classes=W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
      # print(1)
      score=X[i].dot(W)
      score_exp=np.exp(score-np.max(score))
      softmax_prob=score_exp/np.sum(score_exp)
      probability=score_exp[y[i]]/np.sum(score_exp)
      for j in range(num_classes):
          dW[:,j]=dW[:,j]+X[i]*softmax_prob[j]
      dW[:,y[i]]-=X[i]
      loss=loss-np.log(probability)
      
      
    loss/=num_train
    dW/=num_train
    loss=loss+reg*np.sum(W*W)
    dW+= 2*reg*W
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    score=X.dot(W)
    num_train=X.shape[0]
    num_classes=W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    score_exp=np.exp(score-np.max(score))
    softmax_prob=score_exp/np.sum(score_exp,axis=1).reshape(-1,1)
    # print(softmax_prob.shape,y.shape)
    loss+= np.sum(-np.log(softmax_prob[np.arange(len(y)),y]))
    softmax_prob[(np.arange(num_train),y)]-=1
    dW=np.matmul((X.T),softmax_prob)
    
    
    
    loss/=num_train
    dW/=num_train
    loss=loss+reg*np.sum(W*W)
    dW+= 2*reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
