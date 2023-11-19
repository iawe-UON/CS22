from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    num_class=X.shape[0]
    num_train=W.shape[1]
    margin=np.dot(X,W)
    for i in range(num_class):#求loss
      f=margin[i]#W处理过后的数据
      f-=np.max(f)
      p=np.exp(f)/np.sum(np.exp(f))
      loss+=-np.log(p[y[i]])
      dW[:,y[i]]-=X[i]#求导公式推出,梯度是根据不同分类器推出来的
      for j in range(num_train):
        dW[:,j]+=p[j]*X[i]#-X[i]
    loss=loss/num_class+0.5*reg*np.sum(W*W)#切记加上正则化项
    dW=dW/num_class+reg*W
    
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    num_class=X.shape[0]
    num_train=W.shape[1]
    margin=X.dot(W)
    f=margin-margin.max(axis=1).reshape(num_class,1)
    s=np.exp(f).sum(axis=1)
    loss=np.log(s).sum()-f[range(num_class),y].sum()
    loss=loss/num_class+reg*np.sum(W*W)

    #参考答案：
    counts = np.exp(f) / s.reshape(num_class, 1)
    counts[range(num_class), y] -= 1
    dW = np.dot(X.T, counts)

    dW=dW/num_class+reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW
