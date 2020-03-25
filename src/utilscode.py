import sys
import numpy as np
import torch
from scipy.sparse import coo_matrix
import pickle as pkl
import distributions


DTYPE = torch.float32

FIELD_SIZES = [0] * 26
with open('../input/data/featindex.txt') as f:
    for line in f:
        line = line.strip().split(':')
        if len(line) > 1:
            f = int(line[0]) - 1
            FIELD_SIZES[f] += 1
print('field sizes: ', FIELD_SIZES)
FIELD_OFFSETS = [sum(FIELD_SIZES[:i]) for i in range(len(FIELD_SIZES))]
INPUT_DIM = sum(FIELD_SIZES)
OUTPUT_DIM = 1
STDDEV = 1e-3
MINVAL = -1e-3
MAXVAL = 1e-3

def read_data(file_name):
    X = []
    D = []
    y = []
    with open(file_name) as f:
        for line in f:
            fields = line.strip().split()
            y_i = int(fields[0])
            X_i = [int(x.split(':')[0]) for x in fields[1:]]
            D_i = [int(x.split(':')[1]) for x in fields[1:]]
            y.append(y_i)
            X.append(X_i)
            D.append(D_i)
    y = np.reshape(np.array(y), [-1])
    X = libsvm_2_coo(zip(X, D), (len(X), INPUT_DIM)).tocsr()
    
    return X, y


def shuffle(data):
    X, y = data
    ind = np.arange(X.shape[0])
    for i in range(7):
        np.random.shuffle(ind) # 直接在ind上交换
    return X[ind], y[ind]


def libsvm_2_coo(libsvm_data, shape):
    coo_rows = []
    coo_cols = []
    coo_data = []
    n = 0
    for x, d in libsvm_data:
        coo_rows.extend([n] * len(x))
        coo_cols.extend(x)
        coo_data.extend(d)
        n += 1
    coo_rows = np.array(coo_rows)
    coo_cols = np.array(coo_cols)
    coo_data = np.array(coo_data)
    
    return coo_matrix((coo_data, (coo_rows, coo_cols)), shape=shape)


def csr_2_input(csr_mat):
    if not isinstance(csr_mat, list):
        coo_mat = csr_mat.tocoo()
        indices = np.vstack((coo_mat.row, coo_mat.col)).transpose()
        values = csr_mat.data
        shape = csr_mat.shape
        return indices, values, shape
    
    else:
        inputs = []
        for csr_i in csr_mat:
            inputs.append(csr_2_input(csr_i))
        return inputs

def slice(csr_data, start=0, size=-1):
    if not isinstance(csr_data[0], list):
        if size == -1 or start + size >= csr_data[0].shape[0]:
            slc_data = csr_data[0][start: ]
            slc_labels = csr_data[1][start: ]
        else:
            slc_data = csr_data[0][start: start + size]
            slc_labels = csr_data[1][start: start + size]
    else:
        if size == -1 or start + size >= csr_data[0][0].shape[0]:
            slc_data = []
            for d_i in csr_data[0]:
                slc_data.append(d_i[start: ])
            slc_labels = csr_data[1][start: ]
        else:
            slc_data = []
            for d_i in csr_data[0]:
                slc_data.append(d_i[start: start + size])
            slc_labels = csr_data[1][start: start + size]
    return csr_2_input(slc_data), slc_labels


def init_var_map(init_vars, init_path=None):
    if init_path is not None:
        load_var_map = pkl.load(open(init_path, 'rb'))
        print('load variable map from', init_path, load_var_map.keys())
    var_map = {}
    for var_name, var_shape, init_method, dtype in init_vars:
        if init_method == 'zero':
            var_map[var_name] = torch.tensor(torch.zeros(var_shape, dtype=dtype), requires_grad=True, names=var_name, dtype=dtype)
        elif init_method == 'one':
            var_map[var_name] = torch.tensor(torch.ones(var_shape, dtype=dtype), requires_grad=True, names=var_name, dtype=dtype)
        elif init_method == 'normal':
            var_map[var_name] = torch.tensor(torch.normal(var_shape, mean=0.0, std=STDDEV, dtype=dtype), names=var_name, dtype=dtype, requires_grad=True)
        elif init_method == 'tnormal':
            var_map[var_name] = torch.tensor(distributions.truncated_normal_(var_shape, mean=0.0, std=STDDEV, dtype=dtype), names=var_name, dtype=dtype, requires_grad=True)
        elif init_method == 'unifom':
            var_map[var_name] = torch.tensor(torch.empty(var_shape, dtype=dtype).uniform_(MINVAL, MAXVAL), names=var_name, dtype=dtype, requires_grad=True)
        elif init_method == 'xavier':
            maxval = np.sqrt(6. / np.sum(var_shape))
            minval = -maxval
            var_map[var_name] = torch.tensor(torch.empty(var_shape, dtype=dtype).uniform_(minval, maxval), names=var_name, dtype=dtype, requires_grad=True)
        elif isinstance(init_method, int) or isinstance(init_method, float):
            var_map[var_name] = torch.tensor(torch.ones(var_shape, dtype=dtype) * init_method, names=var_name, dtype=dtype)
        elif init_method in load_var_map:
            if load_var_map[init_method].shape == tuple(var_shape):
                var_map[var_name] = torch.tensor(load_var_map[init_method], names=var_shape, dtype=dtype)
            else:
                print('BadParam: init method', init_method, 'shape', var_shape, load_var_map[init_method].shape)
        else:
            print('BadParam: init method', init_method)
    return var_map


def activate(weights, activation_function):
    if activation_function == 'sigmoid':
        return torch.sigmoid(weights)
    elif activation_function == 'softmax':
        return torch.softmax(weights)
    elif activation_function == 'relu':
        return torch.relu(weights)
    elif activation_function == 'tanh':
        return torch.tanh(weights)
    elif activation_function == 'elu':
        return torch.nn.ELU(weights)
    elif activation_function == 'None':
        return weights
    else: 
        return weights


def gather_2d(params, indices):
    shape = params.size()
    flat = torch.reshape(params, (-1, ))
    flat_idx = indices[:, 0] * shape[1] + indices[:, 1]
    flat_idx = torch.reshape(flat, [-1])
    return torch.gather(flat, flat_idx)


def gater_3d(params, indices):
    shape = params.size()
    flat = torch.reshape(params, (-1, ))
    flat_idx = indices[:, 0] * shape[1] * shape[2] + indices[:, 1] * shape[2] + indices[:, 2]
    flat_idx = torch.reshape(flat_idx, [-1])
    return torch.gather(flat, flat_idx)


def gather_4d(params, indices):
    shape = params.size()
    flat = torch.reshape(params, (-1, ))
    flat_idx = indices[:, 0] * shape[1] * shape[2] * shape[3] + indices[:, 1] * shape[2] * shape[3] + indices[:, 2] * shape[3] + indices[:, 3]
    flat_idx = torch.reshape(flat_idx, [-1])
    return torch.gather(flat, flat_idx)


def max_pool_2d(params, k):
    _, indices = torch.topk(params, k, sorted=False)
    shape = indices.size()
    r1 = torch.reshape(torch.range(shape[0]), (-1, 1))
    r1 = r1.repeat(1, k)
    r1 = torch.reshape(r1, (-1, 1))
    indices = torch.cat((r1, torch.reshape(indices, (-1, 1))), 1)
    return torch.reshape(gather_2d(params, indices), (-1, k))


def max_pool_3d(params, k):
    _, indices = torch.topk(params, k, sorted=False)
    shape = torch.size(indices)
    r1 = torch.reshape(torch.range(shape[0]), (-1, 1))
    r2 = torch.reshape(torch.range(shape[1]), (-1, 1))
    r1 = r1.repeat((1, k * shape[1]))
    r2 = r2.repeat((1, k))
    r1 = torch.reshape(r1, (-1, 1))
    r2 = torch.reshape(r2, (-1, 1)).repeat(shape[0], 1)
    indices = torch.cat((r1, r2, torch.reshape(indices, (-1, 1))), 1)
    return torch.reshape(gater_3d(params, indices), (-1, shape[1], k))


def max_pool_4d(params, k):
    _, indices = torch.topk(params, k, sorted=False)
    shape = indices.size()
    r1 = torch.reshape(torch.range(shape[0]), (-1, 1))
    r2 = torch.reshape(torch.range(shape[1]), (-1, 1))
    r3 = torch.reshape(torch.range(shape[2]), (-1, 1))
    r1 = r1.repeat((1, shape[1] * shape[2] * k))
    r2 = r2.repeat((1, shape[2] * k))
    r3 = r3.repeat((1, k))
    r1 = torch.reshape(r1, (-1, 1))
    r2 = torch.reshape(r2, (-1, 1)).repeat(shape[0], 1)
    r3 = torch.reshape(r3, (-1, 1)).repeat(shape[0] * shape[1], 1)
    indices = torch.cat((r1, r2, r3, torch.reshape(indices, (-1, 1))), 1)
    return torch.reshape(gather_4d(params, idices), (-1, shape[1], shape[2], k))