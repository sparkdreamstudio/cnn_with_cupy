import sys
import math
import cupy as cp
import mnist
import random
import numba
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
# General params
VALIDATION_CAPACITY = 2000
MINI_BATCH_SIZE=200
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
EPOCH_NUM = 10
W_INIT = 0.01
IMAGE_SIZE = 28
IMAGE_CHANNEL = 1
CONV1_I_SIZE = 28
CONV1_F_SIZE = 3
CONV1_STRIDES = 1
CONV1_O_SIZE = 28   
CONV1_O_DEPTH = 32

POOL1_F_SIZE = 2
POOL1_STRIDES = 2

CONV2_I_SIZE = 14
CONV2_F_SIZE = 3
CONV2_STRIDES = 1
CONV2_O_SIZE = 14
CONV2_O_DEPTH = 64

POOL2_F_SIZE = 2
POOL2_STRIDES = 2

CONV3_I_SIZE = 7
CONV3_F_SIZE = 3
CONV3_STRIDES = 1
CONV3_O_SIZE = 7
CONV3_O_DEPTH = 128

POOL3_F_SIZE = 2
POOL3_STRIDES = 2

FC1_SIZE_INPUT = 2048
FC1_SIZE_OUTPUT = 625
NN_L_RATE = {1: 0.001, 2: 0.0005, 3: 0.0002, 4: 0.0001, 5: 0.00005, 6: 0.00001, 7: 0.000005,8:0.000001,9:0.0000005,10:0.0000001, 100: 0.000002}
CNN_L_RATE = {1: 0.001, 2: 0.0005, 3: 0.0001, 4: 0.00005, 5: 0.00001, 6: 0.000005, 7: 0.000001,8:0.0000005,9:0.0000001,10:0.00000001, 100: 0.000002}
LEARNING_BASE_RATE = 0.1 
LEARNING_DECAY_RATE = 0.99 

# Pass Activator
class NoAct(object):
    @staticmethod
    def activate(x):
        return x

    @staticmethod
    def bp(delta, x):
        return delta

# ReLU  Activator
class ReLU(object):
    @staticmethod
    def activate(x):
        return np.maximum(0, x)

    @staticmethod
    def bp(delta, x):
        delta[x <= 0] = 0
        return delta

# Sigmoid  Activator
class Sigmoid(object):
    @staticmethod
    def activate(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def bp(delta, x):
        sig = Sigmoid.activate(x)
        return sig * (1 - sig)*delta


class FCLayer(object):
    def __init__(self, i_size, o_size, activator, optimizer,dropout=0.25):
        super().__init__()
        self.w = W_INIT * np.random.randn(i_size, o_size)
        self.b = np.zeros(o_size)
        self.activator = activator
        self.optimizer = optimizer
        self.dppercent=dropout

    def forward(self, layerdata):
        self.dropout(self.dppercent)
        retain_prob=1./(1.-self.dppercent)
        current_Z=cp.asnumpy(cp.dot(cp.asarray(layerdata),cp.asarray(self.w)*cp.asarray(self.dp)*retain_prob)+cp.asarray(self.b))
        self.out = self.activator.activate(current_Z)
        return self.out
    def predict(self, layerdata):
        retain_prob=1./(1.-self.dppercent)
        current_Z=cp.asnumpy(cp.dot(cp.asarray(layerdata), cp.asarray(self.w)*retain_prob)+cp.asarray(self.b))
        return self.activator.activate(current_Z)
    def dropout(self,percent):
        i_size=self.w.shape[0]
        rng=[b for b in range(i_size)]
        drop_num=(int)(i_size*percent)
        rng=np.asarray(rng)
        np.random.shuffle(rng)
        rng=rng[0:drop_num]
        self.dp=np.ones((1,i_size))
        self.dp[:,rng]=0
        self.dp=self.dp.T
    def backpropagation(self, input, deltaLayer, rate):
        self.deltaOri = self.activator.bp(deltaLayer, self.out)
        self.bpDelta()
        self.bpWeights(input, rate)
        self.out=None
        self.deltaOri=None
        return self.deltaPrev

    def bpDelta(self):
        self.deltaPrev = cp.asarray(cp.dot(cp.asarray(self.deltaOri), cp.asarray(self.w).T))

    def bpWeights(self, input, lrt):
        deltaOri_gpu=cp.asarray(self.deltaOri)
        dw = cp.asnumpy(cp.dot(cp.asarray(input).T, deltaOri_gpu))
        db = cp.asnumpy(cp.sum(deltaOri_gpu, axis=0, keepdims=True).reshape(self.b.shape))
        wNew, bNew = self.optimizer.getUpdWeights(self.w, dw*self.dp, self.b, db, lrt)
        self.w = wNew
        self.b = bNew


class ConvLayer(object):
    def __init__(self, i_size, channel, f_size, o_depth, o_size,strides, activator, optimizer):
        super().__init__()
        self.i_size = i_size
        self.channel = channel
        self.f_size = f_size
        self.o_depth = o_depth
        self.o_size = o_size
        self.strides = strides
        self.activator = activator
        self.optimizer = optimizer
        self.w = W_INIT*np.random.randn(o_depth, channel, f_size, f_size)
        self.b = np.zeros((o_depth, 1))
        self.padding = ((self.o_size-1)/strides+f_size-i_size)/2

    def forward(self, prev_A):
        self.out = self.activator.activate(self.conv(
            cp.asarray(self.w), cp.asarray(prev_A), cp.asarray(self.b), self.o_size, self.strides))
        return self.out
    def predict(self, prev_A):
        return self.activator.activate(self.conv(
            cp.asarray(self.w), cp.asarray(prev_A), cp.asarray(self.b), self.o_size, self.strides))

    def conv(self, w, data, b, o_size, strides):
        batchSize = data.shape[0]
        o_depth=w.shape[0]
        f_size=w.shape[2]
        channel=data.shape[1]
        i_size = data.shape[2]
        data_2d = cp.zeros(
            (batchSize, channel*f_size*f_size, o_size*o_size))
        padding_data = data
        pad =int(((o_size - 1) * strides + f_size - i_size) / 2)
        if(pad > 0):
            #padding_data=Math.padding(padding_data,pad)
            padding_data = cp.pad(padding_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)),mode='constant')

        data_2d=self.vectorize_conv(padding_data,f_size,o_size,strides)
        w_2d = cp.reshape(
            w, (o_depth, channel*f_size*f_size))
        result = cp.zeros((batchSize, o_depth, o_size*o_size))
        for row in range(batchSize):
            result[row] = matmul(w_2d, data_2d[row])+b
        result = cp.reshape(
            result, (batchSize, o_depth, o_size,o_size))
        return cp.asnumpy(result)

    def backpropagation(self, prev_A, deltaLayer, rate):
        #dim deltaOri is batch*o_depth*o_size*o_size
        self.deltaOri = self.activator.bp(deltaLayer, self.out)
        self.deltaPrev,dw,db=self.calcurrent_prev_delta(prev_A)
        wNew, bNew = self.optimizer.getUpdWeights(self.w, dw, self.b, db, rate)
        self.w = wNew
        self.b = bNew
        self.deltaOri=None
        self.out=None
        return self.deltaPrev
    def calcurrent_prev_delta(self,prev_A):
        w_rtUD = self.w[:,:,::-1] 
        w_rtLR = w_rtUD[:,:,:,::-1]
        w_rt = w_rtLR.transpose(1,0,2,3)
        prev_delta=self.conv(cp.asarray(w_rt),cp.asarray(self.deltaOri),0,self.o_size,1)
        dw= self.deltawconv(cp.asarray(prev_A))
        db = np.sum(np.sum(np.sum(self.deltaOri, axis=-1), axis=-1), axis=0).reshape(-1, 1)
        return prev_delta,dw,db
    @numba.jit
    def deltawconv(self,pre_A):
        batchSize = pre_A.shape[0]
        o_depth=self.deltaOri.shape[1]
        f_size=self.deltaOri.shape[2]
        channel=pre_A.shape[1]
        o_size=self.f_size
        i_size = pre_A.shape[2]
        strides=1
        pad =int(((o_size- 1) * strides + f_size - i_size) / 2)
        padding_pre_A = pre_A
        if(pad>0):
            padding_pre_A = cp.pad(padding_pre_A, ((
                0, 0), (0, 0), (pad, pad), (pad, pad)),mode='constant')
        data_2d=self.vectorize_convdw(padding_pre_A,f_size,o_size,1)
        deltaOri_2d=cp.reshape(cp.array(self.deltaOri),(batchSize,o_depth,f_size*f_size))
        deltaw=cp.zeros((batchSize,channel,o_depth,o_size*o_size))
        for batch in range(batchSize):
            for c in range(channel):
                deltaw[batch][c]=matmul(deltaOri_2d[batch],data_2d[batch][c])
        deltaw=cp.sum(deltaw,axis=0)
        deltaw=cp.transpose(deltaw,(1,0,2)).reshape(self.w.shape)
        return cp.asnumpy(deltaw)
    @numba.jit
    def vectorize_convdw(self,data,f_size,o_size,strides):
        batchSize=data.shape[0]
        channel=data.shape[1]
        data_2d = cp.zeros(
            (batchSize, channel,f_size*f_size, o_size*o_size))
        for i in range(o_size):
            for j in range(o_size):
                start_h = i*strides
                start_w = j*strides
                data_2d[:,:, :, i*o_size+j] = data[:, :, start_h:start_h+f_size,
                                                           start_w:start_w+f_size].reshape(batchSize, channel,f_size*f_size)
        return data_2d
    @numba.jit
    def vectorize_conv(self,data,f_size,o_size,strides):
        batchSize=data.shape[0]
        channel=data.shape[1]
        data_2d = cp.zeros(
            (batchSize, channel*f_size*f_size, o_size*o_size))
        for i in range(o_size):
            for j in range(o_size):
                start_h = i*strides
                start_w = j*strides
                data_2d[:, :, i*o_size+j] = data[:, :, start_h:start_h+f_size,
                                                           start_w:start_w+f_size].reshape(batchSize, channel*f_size*f_size)
        return data_2d


class MaxPoolLayer(object):
    def __init__(self , f_size,
                 strides, needReshape):

        self.f_size = f_size
        self.strides = strides
        self.needReshape = needReshape
        self.out = []
        self.shapeOfOriOut = ()
        self.poolIdx = []
        self.deltaPrev = []  
        self.deltaOri = [] 

    def forward(self, input):

        pooling, self.poolIdx = self.pool(cp.asarray(input), self.f_size, self.strides, 'MAX')
        self.shapeOfOriOut = pooling.shape
        self.inputShape=input.shape
        if True == self.needReshape:
            self.out = pooling.reshape(pooling.shape[0], -1)
        else:
            self.out = pooling
        return self.out
    def predict(self, input):

        pooling, self.poolIdx = self.pool(cp.asarray(input), self.f_size, self.strides, 'MAX')
        self.shapeOfOriOut = pooling.shape
        if True == self.needReshape:
            return pooling.reshape(pooling.shape[0], -1)
        else:
            return pooling
    def backpropagation(self, input, delta, lrt):

        if True == self.needReshape:
            self.deltaOri = delta.reshape(self.shapeOfOriOut)
        else:
            self.deltaOri = delta

        self.deltaPrev = self.bp4pool(cp.asarray(self.deltaOri), cp.asarray(self.poolIdx),
                                      self.f_size, self.strides, 'MAX')
        self.deltaOri=None
        self.out=None
        return self.deltaPrev

    def pool(self, x, filter_size, strides=2, type='MAX'):

        batches = x.shape[0]
        depth_i = x.shape[1]
        input_size = x.shape[2]  
        x_per_filter = filter_size * filter_size
        output_size = math.ceil((input_size - filter_size) / strides) + 1
        y_per_o_layer = output_size * output_size  
        x_vec = cp.zeros((batches, depth_i, y_per_o_layer, x_per_filter))
        padding_size= (output_size-1)*strides+filter_size
        if(padding_size==input_size):
            padding_x=x
        else:
            padding_x= cp.zeros((batches, depth_i,padding_size,padding_size))-cp.inf
            padding_x[:,:,0:input_size,0:input_size]=x
        for j in range(y_per_o_layer):
            b = int(j / output_size) * strides
            c = (j % output_size) * strides
            x_vec[:, :, j, 0:x_per_filter] = padding_x[:, :, b:b + strides, c:c + strides].reshape(batches, depth_i,
                                                                                           x_per_filter)

        pooling = cp.max(x_vec, axis=3).reshape(batches, depth_i, output_size, output_size)
        pooling_idx = cp.eye(x_vec.shape[3], dtype=int)[x_vec.argmax(3)]


        return cp.asnumpy(pooling), cp.asnumpy(pooling_idx)


    def bp4pool(self, dpool, pool_idx, pool_f_size, pool_strides, type='MAX'):
        batches = dpool.shape[0]
        depth_i = pool_idx.shape[1]
        y_per_o = pool_idx.shape[2]

        x_per_filter = pool_f_size * pool_f_size
        pool_o_size = int(cp.sqrt(y_per_o))

        input_size = (pool_o_size - 1) * pool_strides + pool_f_size
        dpool_reshape = dpool.reshape(batches, depth_i, y_per_o)

        dpool_i_tmp = cp.zeros((batches, depth_i, input_size, input_size))
        pool_idx_reshape = cp.zeros(dpool_i_tmp.shape)
        for j in range(y_per_o):
            b = int(j / pool_o_size) * pool_strides
            c = (j % pool_o_size) * pool_strides
            pool_idx_reshape[:, :, b:b + pool_f_size, c:c + pool_f_size] = pool_idx[:, :, j, 0:x_per_filter].reshape(
                batches,
                depth_i,
                pool_f_size,
                pool_f_size)
            for row in range(pool_f_size):  
                for col in range(pool_f_size):
                    dpool_i_tmp[:, :, b + row, c + col] = dpool_reshape[:, :, j]
        dpool_i = dpool_i_tmp * pool_idx_reshape
        if(self.inputShape[2]<input_size):
            return cp.asnumpy(dpool_i[:,:,0:self.inputShape[2],0:self.inputShape[3]])
        else:
            return cp.asnumpy(dpool_i)
@numba.jit
def matmul(a, b):
    return cp.matmul(a, b)
class Math(object):
    @staticmethod
    def padding(x, pad):

        size_x = x.shape[2]  
        size = size_x + pad * 2 
        if x.ndim == 4:  
            padding = cp.zeros((x.shape[0], x.shape[1], size, size))
            padding[:, :, pad: pad + size_x, pad: pad + size_x] = x

        elif x.ndim == 3: 
            padding = cp.zeros((x.shape[0], size, size))
            padding[:, pad: pad + size_x, pad: pad + size_x] = x

        return padding
    @staticmethod
    def softmax(y): 
        y_cp=cp.asarray(y)
        max_y = cp.max(y_cp, axis=1)
        max_y.shape = (-1, 1)
        y_minus_max = y_cp - max_y
        exp_y = cp.exp(y_minus_max)
        sigma_y = cp.sum(exp_y, axis=1)
        sigma_y.shape = (-1, 1)
        softmax_y = exp_y/sigma_y
        return cp.asnumpy(y_minus_max), cp.asnumpy(softmax_y)

    @staticmethod
    def crossEntropy(y_minus_max, softmax_y, y):
        safe_softmax_y = y_minus_max+np.log(softmax_y)
        result = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            result[i] -= safe_softmax_y[i][y[i]]
        return np.sum(result)


class AdmOptimizer(object):
    def __init__(self, beta1, beta2, eps):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.isInited = False
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []
        self.Iter = 0

    # lazy init
    def initMV(self, shapeW, shapeB):
        if (False == self.isInited):
            self.m_w = np.zeros(shapeW)
            self.v_w = np.zeros(shapeW)
            self.m_b = np.zeros(shapeB)
            self.v_b = np.zeros(shapeB)
            self.isInited = True

    def getUpdWeights(self, w, dw, b, db, lr):
        self.initMV(w.shape, b.shape)

        t = self.Iter + 1
        wNew, self.m_w, self.v_w = self.OptimzAdam(
            cp.asarray(w), cp.asarray(dw), cp.asarray(self.m_w), cp.asarray(self.v_w), lr, t)
        bNew, self.m_b, self.v_b = self.OptimzAdam(
            cp.asarray(b), cp.asarray(db), cp.asarray(self.m_b), cp.asarray(self.v_b), lr, t)
        self.Iter += 1
        return wNew, bNew

    def OptimzAdam(self, x, dx, m, v, lr, t):
        m = self.beta1 * m + (1 - self.beta1) * dx
        mt = m / (1 - self.beta1 ** t)
        v = self.beta2 * v + (1 - self.beta2) * (dx ** 2)
        vt = v / (1 - self.beta2 ** t)
        x += - lr * mt / (cp.sqrt(vt) + self.eps)
        return cp.asnumpy(x), cp.asnumpy(m), cp.asnumpy(v)


class GDOptimizer(object):
    def __init__(self):
        super().__init__()
        self.Iter=0
    def getUpdWeights(self, w, dw, b, db, lr):
        self.Iter+=1
        w -= lr*dw
        b -= lr*db
        return w, b
class ResultView(object):
    def __init__(self, epoch, line_labels, colors, ax_labels, dataType):
        self.cur_p_idx = 0
        self.curv_x = np.zeros(epoch * 100, dtype=int)
        self.curv_ys = np.zeros((4, epoch * 100), dtype=dataType)
        self.line_labels = line_labels
        self.colors = colors
        self.ax_labels = ax_labels

    def addData(self, curv_x, loss, loss_v, acc, acc_v):

        self.curv_x[self.cur_p_idx] = curv_x
        self.curv_ys[0][self.cur_p_idx] = loss
        self.curv_ys[1][self.cur_p_idx] = loss_v
        self.curv_ys[2][self.cur_p_idx] = acc
        self.curv_ys[3][self.cur_p_idx] = acc_v
        self.cur_p_idx += 1

    def save(self,name,_1_y_l=0.1,_2_y_l=0.02):
        self.savefig(name,self.cur_p_idx, self.curv_x, self.curv_ys, self.line_labels, self.colors, self.ax_labels,_1_y_l,_2_y_l)

    def savefig(self,name, idx, x, ys, line_labels, colors, ax_labels,_1_y_l,_2_y_l):
        LINEWIDTH = 2.0
        plt.figure(figsize=(8, 4))
        # loss
        ax1 = plt.subplot(211)
        for i in range(2):
            line = plt.plot(x[:idx], ys[i][:idx])[0]
            plt.setp(line, color=colors[i], linewidth=LINEWIDTH, label=line_labels[i])

        ax1.xaxis.set_major_locator(MultipleLocator(300))
        ax1.yaxis.set_major_locator(MultipleLocator(_1_y_l))
        ax1.set_xlabel(ax_labels[0])
        ax1.set_ylabel(ax_labels[1])
        plt.grid()
        plt.legend()

        ax2 = plt.subplot(212)
        for i in range(2, 4):
            line = plt.plot(x[:idx], ys[i][:idx])[0]
            plt.setp(line, color=colors[i], linewidth=LINEWIDTH, label=line_labels[i])

        ax2.xaxis.set_major_locator(MultipleLocator(300))
        ax2.yaxis.set_major_locator(MultipleLocator(_2_y_l))
        ax2.set_xlabel(ax_labels[0])
        ax2.set_ylabel(ax_labels[2])

        plt.grid()
        plt.legend()
        plt.savefig(name)

class Recognition(object):
    def __init__(self):
        super().__init__()
        self.layers = []

    def forward(self, traindata, y):
        current_a = traindata
        for layer in self.layers:
            current_a = layer.forward(current_a)
        y_ = np.argmax(current_a, axis=1)
        acc_t = np.mean(y == y_)

        y_minus_max, softmax_y = Math.softmax(current_a)
        loss = Math.crossEntropy(y_minus_max, softmax_y, y)/y.shape[0]

        softmax_y[range(softmax_y.shape[0]), y] -= 1
        delta = softmax_y / softmax_y.shape[0]
        return acc_t, loss, delta
    def predict(self,data,y):
        current_a=data
        for layer in self.layers:
            current_a = layer.predict(current_a)
            cp._default_memory_pool.free_all_blocks()
        y_ = np.argmax(current_a, axis=1)
        y_minus_max, softmax_y = Math.softmax(current_a)
        loss = Math.crossEntropy(y_minus_max, softmax_y, y)/y.shape[0]
        return y_,loss
    def backpropagation(self,traindata, delta, rate):
        deltaLayer = delta
        for i in reversed(range(1, len(self.layers))):
            deltaLayer = self.layers[i].backpropagation(
                self.layers[i-1].out, deltaLayer, rate)  
        self.layers[0].backpropagation(traindata, deltaLayer, rate)

    def train_steps(self, train_data, y, rate):
        acc, loss, delta = self.forward(train_data, y)
        self.backpropagation(train_data,delta, rate)
        return acc, loss

    def validation(self, data_v, y_v):
        result,loss= self.predict(data_v,y_v)
        return np.mean(result == y_v),result,loss
def getTrainRanges(data, miniBatchSize):
    r = [i for i in range(len(data))]
    random.shuffle(r)
    rngs = [r[i:i + miniBatchSize] for i in range(0, len(r), miniBatchSize)]
    return rngs
def sample_validation(data,y,validation_size):
    samples_v = random.sample([i for i in range(len(data))], validation_size)
    x = np.zeros((validation_size,data.shape[1],data.shape[2],data.shape[3]))
    y_=np.zeros((validation_size),dtype=int)
    for idx,sample in enumerate(samples_v):
        x[idx]=data[sample]
        y_[idx]=y[sample]
    return x,y_
def cnn_adm(view):
    
    conv1Optimizer = AdmOptimizer(BETA1, BETA2, EPS)
    conv1 = ConvLayer(IMAGE_SIZE, IMAGE_CHANNEL,
                      CONV1_F_SIZE, CONV1_O_DEPTH,
                      CONV1_O_SIZE, CONV1_STRIDES,
                      ReLU, conv1Optimizer)
    pool1 = MaxPoolLayer(POOL1_F_SIZE,
                         POOL1_STRIDES, False)

    conv2Optimizer = AdmOptimizer(BETA1, BETA2, EPS)
    conv2 = ConvLayer(CONV2_I_SIZE, CONV1_O_DEPTH,
                      CONV2_F_SIZE, CONV2_O_DEPTH,
                      CONV2_O_SIZE, CONV2_STRIDES,
                      ReLU, conv2Optimizer)
    pool2 = MaxPoolLayer( POOL2_F_SIZE,
                         POOL2_STRIDES, False)
    conv3Optimizer = AdmOptimizer(BETA1, BETA2, EPS)
    conv3 = ConvLayer(CONV3_I_SIZE, CONV2_O_DEPTH,
                      CONV3_F_SIZE, CONV3_O_DEPTH,
                      CONV3_O_SIZE, CONV3_STRIDES,
                      ReLU, conv3Optimizer)
    pool3 = MaxPoolLayer( POOL3_F_SIZE,
                         POOL3_STRIDES, True)
    fc1Optimizer = AdmOptimizer(BETA1, BETA2, EPS)
    fc1 = FCLayer(FC1_SIZE_INPUT, FC1_SIZE_OUTPUT, ReLU,
                  fc1Optimizer) 
    fc2Optimizer = AdmOptimizer(BETA1, BETA2, EPS)
    fc2 = FCLayer(FC1_SIZE_OUTPUT, 10, NoAct,
                  fc2Optimizer)

    session = Recognition()
    session.layers=[conv1, pool1, conv2, pool2,conv3, pool3, fc1, fc2]
    train_labels = mnist.train_labels()
    train_image = mnist.train_images()/255
    train_image = np.reshape(train_image,(train_image.shape[0],1,train_image.shape[1],train_image.shape[2]))
    test_labels = mnist.test_labels()
    test_image = mnist.test_images()/255
    test_image = np.reshape(test_image,(test_image.shape[0],1,test_image.shape[1],test_image.shape[2]))
    for epoch in range(EPOCH_NUM):
        for key in CNN_L_RATE.keys():
            if (epoch + 1) < key:
                break
            learning_rate = CNN_L_RATE[key]
        dataRngs = getTrainRanges(train_labels,MINI_BATCH_SIZE)
        for batch in range(len(dataRngs)):
            x = np.zeros((MINI_BATCH_SIZE,train_image.shape[1],train_image.shape[2],train_image.shape[3]))
            y_=np.zeros((MINI_BATCH_SIZE),dtype=int)
            for idx,sample in enumerate(dataRngs[batch]):
                x[idx]=train_image[sample]
                y_[idx]=train_labels[sample]
            acc_t, loss_t = session.train_steps(
                x, y_, learning_rate)
            print("training----epoch:", epoch,",batch:",batch, ",accuration:", acc_t)
            if (batch % 30 == 0 and (batch+epoch) >0):
                cp._default_memory_pool.free_all_blocks()
                total_result=None
                loss_v=0
                for i in range(5):
                    x_v=test_image[i*2000:i*2000+2000]
                    y_v=test_labels[i*2000:i*2000+2000]
                    _,result,_loss = session.validation(x_v, y_v)
                    loss_v+=_loss
                    if total_result is None:
                        total_result=result
                    else:
                        total_result=np.concatenate([total_result,result])
                acc_v=np.mean(total_result == test_labels)
                loss_v/=5
                print("validation----epoch:", epoch,",batch:",batch, ",accuration:", acc_v)
                cp._default_memory_pool.free_all_blocks()
                view.addData(fc1Optimizer.Iter,
                                 loss_t, loss_v, acc_t, acc_v)
    view.save("cnn_Adam.png",0.1,0.04)
def nn_adm(view):
    fc1Optimizer = AdmOptimizer(BETA1, BETA2, EPS)
    fc1 = FCLayer(784, 300, ReLU,
                  fc1Optimizer) 
    fc2Optimizer = AdmOptimizer(BETA1, BETA2, EPS)
    fc2 = FCLayer(300, 10, NoAct,
                  fc2Optimizer)

    session = Recognition()
    session.layers=[fc1, fc2]
    train_labels = mnist.train_labels()
    train_image = mnist.train_images()/255
    train_image = train_image.reshape(
        train_image.shape[0], train_image.shape[1]*train_image.shape[2])
    test_labels = mnist.test_labels()
    test_image = mnist.test_images()/255
    test_image = test_image.reshape(test_image.shape[0], test_image.shape[1]*test_image.shape[2])
    for epoch in range(EPOCH_NUM):
        for key in NN_L_RATE.keys():
            if (epoch + 1) < key:
                break
            learning_rate = NN_L_RATE[key]*10
        dataRngs = getTrainRanges(train_labels,MINI_BATCH_SIZE)
        for batch in range(len(dataRngs)):
            x = np.zeros((MINI_BATCH_SIZE,train_image.shape[1]))
            y_=np.zeros((MINI_BATCH_SIZE),dtype=int)
            for idx,sample in enumerate(dataRngs[batch]):
                x[idx]=train_image[sample]
                y_[idx]=train_labels[sample]
            acc_t, loss_t = session.train_steps(
                x, y_, learning_rate)
            if (batch % 30 == 0 and (batch+epoch) >0):
                cp._default_memory_pool.free_all_blocks()
                total_result=None
                loss_v=0
                for i in range(5):
                    x_v=test_image[i*2000:i*2000+2000]
                    y_v=test_labels[i*2000:i*2000+2000]
                    _,result,_loss = session.validation(x_v, y_v)
                    loss_v+=_loss
                    if total_result is None:
                        total_result=result
                    else:
                        total_result=np.concatenate([total_result,result])
                acc_v=np.mean(total_result == test_labels)
                loss_v/=5
                print("validation----epoch:", epoch,",batch:",batch, ",accuration:", acc_v)
                cp._default_memory_pool.free_all_blocks()
                view.addData(fc1Optimizer.Iter,
                                 loss_t, loss_v, acc_t, acc_v)
    view.save("nn_Adam.png")
def nn_sgd(view):
    fc1Optimizer = GDOptimizer()
    fc1 = FCLayer(784, 300, ReLU,
                  fc1Optimizer) 
    fc2Optimizer = GDOptimizer()
    fc2 = FCLayer(300, 10, NoAct,
                  fc2Optimizer)

    session = Recognition()
    session.layers=[fc1, fc2]
    train_labels = mnist.train_labels()
    train_image = mnist.train_images()/255
    train_image = train_image.reshape(
        train_image.shape[0], train_image.shape[1]*train_image.shape[2])
    test_labels = mnist.test_labels()
    test_image = mnist.test_images()/255
    test_image = test_image.reshape(test_image.shape[0], test_image.shape[1]*test_image.shape[2])
    for epoch in range(EPOCH_NUM):
        learning_rate = LEARNING_BASE_RATE * (LEARNING_DECAY_RATE**epoch)
        dataRngs = getTrainRanges(train_labels,MINI_BATCH_SIZE)
        for batch in range(len(dataRngs)):
            x = np.zeros((MINI_BATCH_SIZE,train_image.shape[1]))
            y_=np.zeros((MINI_BATCH_SIZE),dtype=int)
            for idx,sample in enumerate(dataRngs[batch]):
                x[idx]=train_image[sample]
                y_[idx]=train_labels[sample] 
            acc_t, loss_t = session.train_steps(
                x, y_, learning_rate)
            if (batch % 30 == 0 and (batch+epoch) >0):
                cp._default_memory_pool.free_all_blocks()
                total_result=None
                loss_v=0
                for i in range(5):
                    x_v=test_image[i*2000:i*2000+2000]
                    y_v=test_labels[i*2000:i*2000+2000]
                    _,result,_loss = session.validation(x_v, y_v)
                    loss_v+=_loss
                    if total_result is None:
                        total_result=result
                    else:
                        total_result=np.concatenate([total_result,result])
                acc_v=np.mean(total_result == test_labels)
                loss_v/=5
                print("validation----epoch:", epoch,",batch:",batch, ",accuration:", acc_v,",loss:",loss_v)
                cp._default_memory_pool.free_all_blocks()
                view.addData(fc1Optimizer.Iter,
                                 loss_t, loss_v, acc_t, acc_v)
    view.save("nn_sgd.png",0.2,0.05)

if __name__ == "__main__":
    view = ResultView(EPOCH_NUM,
                          ['train_loss', 'val_loss', 'train_acc', 'val_acc'],
                          ['y', 'r', 'g', 'b'],
                          ['Iteration', 'Loss', 'Accuracy'],
                          np.float32)
    cnn_adm(view)
    #nn_sgd(view)
    #nn_adm(view)