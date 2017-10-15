import numpy as np

def nonlin(x,deriv=False):
    if(deriv==True):
        return ((x)*(1-(x)))/3
    return 1/(1+np.exp(-x))

def fmax(x,deriv=False):
    if(deriv==True):
        return 0.33
    return np.maximum(x,0)/3


class NN:
    def __init__(self, shapes, func=nonlin):
        self.func = func
        self.shapes = shapes
        self.syns = [ 2*np.random.random((shapes[i-1][1],shapes[i][0])) - 1
                      for i in range(1, len(shapes)) ]
        self.layers = [ np.zeros(shapes[i])
                        for i in range(1, len(shapes)) ]
    
    def learn(self, X, y, cycles):
        for j in range(cycles):
            res = self.calc(X)
            prev = y - res
            for i in range(len(self.layers)-1,-1,-1):
                l_delta = (prev*self.func(self.layers[i], True)).T
                if i == 0:
                    self.syns[i] += X.T.dot(l_delta)
                else:
                    prev = l_delta.dot(self.syns[i].T)
                    self.syns[i] += self.layers[i-1].T.dot(l_delta)
        return self.layers[-1]

    def calc(self,X):
        for i in range(len(self.syns)):
            if i == 0:
                self.layers[i] = self.func(np.dot(X,self.syns[i])).T
            else:
                self.layers[i] = self.func(np.dot(self.layers[i-1],self.syns[i])).T
        return self.layers[-1]


if __name__ == '__main__':
    X = np.array([  [0,0,1],[0,1,1],[1,0,1],[1,1,1]  ])
    y = np.array([[0,1,1,0]]).T

    nn = NN([X.shape, (y.shape[1], X.shape[0]), y.shape ])
    nn.learn(X,y,1000)
    print("X =", X)
    print("y =", y)
    print("Result =", nn.calc(X))
