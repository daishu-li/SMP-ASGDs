import sys
from sklearn.preprocessing import StandardScaler
sys.path.append('C:\\Users\\libs\\PycharmProjects\\project202121\\SGD-SVM\\PSGDSVMPY')
import numpy as np
import pandas as pd
import time
from mpi4py import MPI


class Communication:
    mpi = MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    def __init__(self):
        rank = self.comm.Get_rank()
        if rank == 0:
            print("Initial Configurations : World Size = " + str(self.size))

    def allreduce(self, input=np.empty(1, dtype='i'), output=np.empty(1, dtype='i'), dtype=mpi.Datatype, op=mpi.SUM):
        self.comm.Allreduce([input, dtype], [output, dtype], op=op)


def load_all_data(filename):
    df_chunk = pd.read_csv(filename, chunksize=100)
    res_chunk = []
    for chunk in df_chunk:
        res_chunk.append(chunk)
    data = pd.concat(res_chunk)
    data = data.values
    y = data[:, -1]
    # y=data[:,0]
    # # y[y == 1] = 1
    y[y == 3] = -1
    y[y == 2] = 1
    x = np.delete(data, -1, axis=1)
    del data
    return x,y


datasets = ['hog_data']
n_features = [121104]

# for dataset, features in zip(datasets, n_features):
file = r'C:\Users\libs\PycharmProjects\project202121\SGD-SVM\mat2np\hog_data_8.csv'
# file = r'C:\Users\libs\PycharmProjects\project202121\SGD-SVM\mat2np\hog_data_1.csv'
    # file = r'C:\Users\libs\PycharmProjects\project202121\SGD-SVM\mat2np\dwt.csv'
    # file = r'C:\Users\libs\PycharmProjects\project202121\SGD-SVM\diabetes_scale.csv'
    # test_file = r'C:\Users\libs\PycharmProjects\project202121\SGD-SVM\a1a_test.csv'
x_all, y_all = load_all_data(file)
print(x_all.shape, y_all.shape)
print(np.unique(y_all))
ratio = 0.8
world_size = len(x_all)
split_index = int(world_size * ratio)
x_training = x_all[:split_index]
x_testing = x_all[split_index:]
y_training = y_all[:split_index]
y_testing = y_all[split_index:]
del x_all
del y_all
print(x_training.shape)
print(x_testing.shape)
X = x_training
y = y_training
del x_training
# del x_testing
DATA_TYPE = 'float32'

X = X.astype(DATA_TYPE)
y = y.astype(DATA_TYPE)
n = len(X)
comms = Communication()
rank = comms.comm.Get_rank()
world_size = comms.comm.Get_size()
partition_size = n / world_size
start = int(rank * partition_size)
end = int(start + partition_size)
X_p = X[start:end, :]
y_p = y[start:end]
m1 = len(X_p[0])
print(X_p.shape)

T = 200
if world_size > 1:
    epochs = 200
else:
    epochs = T
print("World Size : ", world_size, epochs)
m1 = len(X[0])
    # w = np.random.uniform(0, 1, m1).astype(DATA_TYPE)
# for i in range(5):
w = np.zeros(m1, DATA_TYPE)
    # w_r = np.random.uniform(0, 1, m1, DATA_TYPE)
w_r = np.zeros(m1, DATA_TYPE)
epsilon = 0.0000001
r = np.zeros(w.shape, DATA_TYPE)
y_preds = []
cost = 1000000
tolerance = 0.01
epoch = 1
gamma = 0.8
C=1
accs = []
costs = []
exp_time = 0
exp_time -= time.time()
    # alpha = 0.01
alpha = 0.0001
    # r = np.zeros(w.shape)
num_range = len(X_p)
indices = np.random.choice(num_range, num_range, replace=False)
while (cost > 0.01 and epoch < epochs):
    gradient = 0
    coefficient = 1
    for index in indices:
        condition = y[index] * np.dot(X[index], w)
        if (condition < 1):
            gradient = alpha * (-(X[index] * y[index]) + (coefficient * w))
        else:
            gradient = alpha * (coefficient * w)
        r = r + np.multiply(gradient, gradient)
        r = r + epsilon
        d1 = np.multiply(gradient, 1.0 / r)
        w = w - (alpha * d1)
    cost = abs(0.5 * np.dot(w, w.T) + C * condition)

    comms.allreduce(input=w, output=w_r, op=comms.mpi.SUM, dtype=comms.mpi.FLOAT)
    w = w_r / world_size
        # cost = abs(0.5 * np.dot(w, w.T) + C * condition)
        # print(cost)
        # costs.append(cost)
    epoch += 1
    labels = []
    for x in x_testing:
        label = np.sign(np.dot(w.T, x))
        labels.append(label)

    y_pred = np.array(labels)
    y_preds.append(y_pred)
    correct = (y_pred == y_testing).sum()
    total = len(y_pred)
    acc = float(correct) / float(total) * 100.0
    print(acc)
    accs.append(acc)
exp_time += time.time()
print(" cost, Acc : ", cost, max(accs))
print("Time : ", exp_time)

