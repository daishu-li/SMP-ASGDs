import sys
import numpy as np
import pandas as pd
# import time
from mpi4py import MPI

sys.path.append('C:\\Users\\libs\\PycharmProjects\\project202121\\SGD-SVM\\PSGDSVMPY')
from sklearn.metrics import precision_score, confusion_matrix, accuracy_score


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
    '''
    1:1       2:1     3:1
    2:-1      1:-1    2:-1
    3:-1      3:-1    1:-1      
    '''
    # y[y == 3] = -1
    # y[y == 2] = -1
    x = np.delete(data, -1, axis=1)
    del data
    return x, y


def cal_metrics(confusion_matrix):
    n_classes = confusion_matrix.shape[0]
    metrics_result = []
    for i in range(n_classes):
        # 逐步获取 真阳，假阳，真阴，假阴四个指标，并计算三个参数
        ALL = np.sum(confusion_matrix)
        # 对角线上是正确预测的
        TP = confusion_matrix[i, i]
        # 列加和减去正确预测是该类的假阳
        FP = np.sum(confusion_matrix[:, i]) - TP
        # 行加和减去正确预测是该类的假阴
        FN = np.sum(confusion_matrix[i, :]) - TP
        # 全部减去前面三个就是真阴
        TN = ALL - TP - FP - FN
        metrics_result.append([TP / (TP + FP), TP / (TP + FN), TN / (TN + FP)])
    return metrics_result


def PSGD(X, y, x_testing, y_testing, world_size, rank, comms):
    print('the shape of trainSet is ', X.shape)
    print('the shape of testSet is ', x_testing.shape)
    DATA_TYPE = 'float32'

    X = X.astype(DATA_TYPE)
    y = y.astype(DATA_TYPE)
    n = len(X)
    indices = np.arange(0, n, 1)
    np.random.shuffle(indices)
    # data_per_machine = n / world_size
    batch_size = 64
    data_per_machine_indices = np.array_split(indices, world_size)
    index_set_of_machine = data_per_machine_indices[rank]
    # indices = index_set_of_machine

    indices = np.random.choice(index_set_of_machine, batch_size, replace=False)
    m1 = len(X[0])

    print('the length of rank {} is {} '.format(rank, X.shape[0]))
    T = 100
    if world_size > 1:
        epochs = 200
    else:
        epochs = T
    print("World Size : ", world_size)
    w = np.zeros(m1, DATA_TYPE)
    w_r = np.zeros(m1, DATA_TYPE)

    # gradient = np.zeros(m1,DATA_TYPE)
    # gradient_r = np.zeros((m1, 3), DATA_TYPE)
    # isComplete = False
    # v = np.zeros(w.shape, DATA_TYPE)
    # m = np.zeros(w.shape, DATA_TYPE)
    # y_preds = []
    # print(m1)
    # print(w.shape[1])

    exp_time = 0
    lam = 2
    # exp_time -= time.time()
    exp_time -= comms.mpi.Wtime()
    for epoch in range(1, epochs):
        alpha = 1.0 / (lam * float(epoch))
        for index in indices:
            # _class = int(y[index])
            condition = y[index] * np.dot(X[index], w)
            # print(condition.shape)
            if condition < 1:
                gradient = -1 * (X[index] * y[index]) + (lam * w)
            else:
                gradient = lam * w
            # w = w - alpha * np.multiply((m_hat), eta/(np.sqrt(v_hat) + epsilon))
            # w[:, _class] = w[:, _class] - alpha * gradient[:, _class]
            w = w - alpha * gradient
            comms.allreduce(input=w, output=w_r, op=comms.mpi.SUM, dtype=comms.mpi.FLOAT)
            w = w_r / world_size
            # comms.bcast(input=w, dtype=comms.mpi.FLOAT, root=0)
        labels = []
        for x in x_testing:
            # label = np.sign(np.dot(w.T, x))
            label = np.sign(np.dot(w.T, x))
            labels.append(label)

        y_pred = np.array(labels)
        # y_preds.append(y_pred)
        correct = (y_pred == y_testing).sum()
        total = len(y_pred)
        # acc2 = float(correct) / float(total) * 100.0
        acc = accuracy_score(y_testing, y_pred)
        # precision = precision_score(y_testing, y_pred)
        # recall =
        cm = confusion_matrix(y_testing, y_pred)
        # print(cm)

        tpr = cm[0][0]/np.sum(cm[0])
        tnr = cm[-1][-1]/np.sum(cm[1])
        precision = cm[0][0]/np.sum(cm[:,0])
        # results = cal_metrics(cm)
        # print('epoch: {}, accuracy: {}'.format(
        #     epoch,
        #     acc))
        # print(acc2)
        print('epoch: {}, accuracy: {}, precision: {}, sensitivity: {}, specificity: {} '.format(
            epoch,
            acc,
            precision,
            tpr,
            tnr
        ))
    # if rank == 0 and epoch == T - 1:
    #     isComplete = True
    exp_time += comms.mpi.Wtime()


# if __name__ == '__main__':
    # file = r'C:\Users\libs\PycharmProjects\project202121\Grade_class\trainSet.csv'
    # file1 = r'C:\Users\libs\PycharmProjects\project202121\SGD-SVM\mat2np\hog_data_label.csv'
file = r'C:\Users\libs\PycharmProjects\project202121\SGD-SVM\mat2np\hog_data_8.csv'
# file = r'C:\Users\libs\PycharmProjects\project202121\SGD-SVM\diabetes_scale.csv'
# test_file = r'C:\Users\libs\PycharmProjects\project202121\SGD-SVM\a1a_test.csv'
# file = r'C:\Users\libs\PycharmProjects\project202121\SGD-SVM\mat2np\dwt.csv'
# n_features = features
# x_training, y_training = load_all_data(train_file)
# x_testing, y_testing = load_all_data(test_file)
x_all, y_all = load_all_data(file)
y = y_all.copy()
print(x_all.shape, y_all.shape)
print(np.unique(y_all))
ratio = 0.8
data_size = x_all.shape[0]
split_index = int(data_size * ratio)
x_training = x_all[:split_index]
x_testing = x_all[split_index:]
# y_copy_1 = y_all.copy()
# y_copy_2 = y_all.copy()
# y_copy_3 = y_all.copy()
y_training1 = y_all[:split_index].copy()
y_training1[y_training1 > 1] = -1
y_testing1 = y_all[split_index:].copy()
y_testing1[y_testing1 > 1] = -1
y_all = y.copy()
y_training2 = y_all[:split_index]
y_training2[y_training2 != 2] = -1
y_training2[y_training2 == 2] = 1

y_testing2 = y_all[split_index:]
y_testing2[y_testing2 != 2] = -1
y_testing2[y_testing2 == 2] = 1
#
#
#
#
y_all = y.copy()
y_training3 = y_all[:split_index]
y_training3[y_training3 != 3] = -1
y_training3[y_training3 == 3] = 1
y_testing3 = y_all[split_index:]
y_testing3[y_testing3 != 3] = -1
y_testing3[y_testing3 == 3] = 1
#
# # n = X.shape[0]
comms = Communication()
rank = comms.comm.Get_rank()
world_size = comms.comm.Get_size()
#
#
PSGD(x_training, y_training1, x_testing, y_testing1, world_size, rank, comms)
PSGD(x_training, y_training2, x_testing, y_testing2, world_size, rank, comms)
PSGD(x_training, y_training3, x_testing, y_testing3, world_size, rank, comms)
    # partition_size = n / world_size
    # start = int(rank * partition_size)
    # end = int(start + partition_size)
    # X_p = X[start:end, :]
    # y_p = y[start:end]

    # print('epoch: {}, accuracy: {}, precision: {}, sensitivity: {}, specificity: {} '.format(
    #     epoch,
    #     acc,
    #     results[0],
    #     results[1],
    #     results[2]
    # ))
    # print('classification report',)
    # accs.append(acc)

    # cost = abs(0.5 * np.dot(w, w.T) + C * condition)
    # costs.append(cost)

    # exp_time += time.time()
    # if (rank == 0 and epoch == epochs - 1):
    # #     isComplete = True
    # if isComplete:
    #     # if rank == 0 and isComplete:
    #     labels = []
    #     for x in x_testing:
    #         label = np.sign(np.dot(w.T, x))
    #         labels.append(label)
    #
    #     y_pred = np.array(labels)
    #     y_preds.append(y_pred)
    #     correct = (y_pred == y_testing).sum()
    #     total = len(y_pred)
    #     acc = float(correct) / float(total) * 100.0
    #     print("Acc : ", acc)
    #     print("Time : ", exp_time)
    #     print()
