import sys

# sys.path.append('C:\\Users\\libs\\PycharmProjects\\project202121\\SGD-SVM\\PSGDSVMPY')
import numpy as np
import pandas as pd
import time
from mpi4py import MPI
# from sklearn.preprocessing import StandardScaler


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
    # y[y == 1] = 1
    y[y == 3] = -1
    y[y == 2] = 1
    x = np.delete(data, -1, axis=1)
    del data
    return x, y


datasets = ['hog_data']
splits = [True]
n_features = [24336]

for dataset, features in zip(datasets, n_features):
    file = r'C:\Users\libs\PycharmProjects\project202121\SGD-SVM\mat2np\hog_data_8.csv'
    # file = r'C:\Users\libs\PycharmProjects\project202121\SGD-SVM\mat2np\hog_data_1.csv'
    # file1 = r'C:\Users\libs\PycharmProjects\project202121\SGD-SVM\mat2np\hog_data_label.csv'
    # file = r'C:\Users\libs\PycharmProjects\project202121\SGD-SVM\diabetes_scale.csv'
    # test_file = r'C:\Users\libs\PycharmProjects\project202121\SGD-SVM\a1a_test.csv'
    # file = r'C:\Users\libs\PycharmProjects\project202121\SGD-SVM\mat2np\dwt.csv'
    # n_features = features
    # x_training, y_training = load_all_data(train_file)
    # x_testing, y_testing = load_all_data(test_file)
    x_all, y_all = load_all_data(file)
    # scale = StandardScaler()
    # x_all = scale.fit_transform(x_all)
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

    T = 100
    if world_size > 1:
        epochs = 100
    else:
        epochs = T
    print("World Size : ", world_size, epochs)
    # m1 = len(X[0])
    w = np.random.uniform(0, 1, m1).astype(DATA_TYPE)
    # w = np.zeros(m1, DATA_TYPE)
    w_r = np.zeros(m1, DATA_TYPE)
    gradient = np.zeros(m1, DATA_TYPE)
    # gradient_r = np.zeros(m1, DATA_TYPE)
    isComplete = False
    v = np.zeros(w.shape, DATA_TYPE)
    m = np.zeros(w.shape, DATA_TYPE)
    beta1_range = [0.5, 0.6, 0.75,0.8, 0.85]
    beta2_range = [0.75,0.8, 0.85, 0.90, 0.93, 0.95, .99,]
    # beta1_range = [.9]
    # beta2_range = [.99]
    y_preds = []
    indices = np.random.choice(nn, nn, replace=False)
    for beta1 in beta1_range:
        for beta2 in beta2_range:
            epsilon = 0.0001
            exp_time = 0
            lam = 2
            exp_time -= time.time()
            accs = []
            costs = []
            eta = 0.1
            C = 1
            nn = len(X_p)

            for epoch in range(1, epochs):
                # alpha = 1.0 / (lam * float(epoch))
                alpha = 1.0 / (1.0 + float(epoch))
                coefficient = ((1 / float(epoch)))

                for index in indices:
                    condition = y_p[index] * np.dot(X_p[index], w)

                    if (condition < 1):
                        # gradient = (-1*(X_p[index] * y[index]) + (lam * w))
                        gradient = alpha * (-(X[index] * y[index]) + (coefficient * w))
                        # gradient = -(X[index] * y[index]) + (coefficient * w)
                    else:
                        gradient = (lam * w)
                        # gradient = alpha * (coefficient * w)
                    m = beta1 * m + (1 - beta1) * gradient
                    m_hat = m / (1 - beta1 ** epoch)
                    grad_mul = (np.multiply(gradient, gradient))
                    v = beta2 * v + (1 - beta2) * grad_mul
                    v_hat = v / (1 - beta2 ** epoch)
                    # eta_t = eta/np.sqrt(epoch)
                    w = w - np.multiply((m_hat), eta / (np.sqrt(v_hat) + epsilon))
                    # w = w - alpha * np.multiply((m_hat), 1 / (np.sqrt(v_hat) + epsilon))
                    comms.allreduce(input=w, output=w_r, op=comms.mpi.SUM, dtype=comms.mpi.FLOAT)
                    w = w_r / world_size
                # cost = abs(0.5 * np.dot(w, w.T) + C * condition)
                # costs.append(cost)
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
            if (rank == 0 and epoch == epochs - 1):
                isComplete = True
            if (isComplete):
                print(" Beta1, Beta2, Acc : ", beta1, beta2, max(accs))
                print("Time : ", exp_time)
