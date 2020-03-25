import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from time import time
import pdb
from torch.utils.data import DataLoader
dtype = torch.float32

# class Model:
#     def __init__(self):
#         self.X = None
#         self.y = None
#         self.layer_keeps = None
#         self.vars = None
#         self.keep_prob_train = None
#         self.keep_prob_test = None
    
#     def run(self, X, y, epochs=EPOCHS):
#         for e in range(epochs):
#             for 

class PNN1(nn.Module):
    def __init__(self, field_sizes=None, feature_sizes=None, embed_size=10, deep_layers=[32, 32, 32], 
                h_depth=3, layer_acts=None, dropout_deep=[0.5, 0.5, 0.5], embed_l2=None, layer_l2=None, 
                init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=950104, 
                deep_layers_acticvation="relu", n_epochs=64, batch_size=256, optimizer_type='adam', 
                loss_type='logloss', eval_metric=roc_auc_score, n_class=1, greater_is_better=None):
        super(PNN1, self).__init__()
        # self.init_vars = []
        # self.init_path = init_path
        # self.num_inputs = len(field_sizes)
        # for i in range(self.num_inputs):
        #     self.init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))
        # num_paris = int(self.num_inputs * (self.num_inputs - 1) / 2)
        # node_in = self.num_inputs * embed_size + num_paris
        # for i in range(len(layer_sizes)):
        #     self.init_vars.append(('w%d' % i, [node_in, layer_sizes[i]], 'xavier', dtype))
        #     self.init_vars.append(('b%d' % i, [layer_sizes[i]], 'zero', dtype))     
        # self.w0 = [self.vars['embed_%d' % i] for i in range(self.num_inputs)]
        # self.Xw = torch.cat(())
        # self.drop_out = drop_out
        # self.embed_size = embed_size
        self.field_size = field_sizes
        self.feature_sizes = feature_sizes
        self.embedding_size = embed_size
        self.h_depth = h_depth
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_acticvation
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.n_class = n_class
        self.greater_is_better = greater_is_better
        
        torch.manual_seed(self.random_seed)
        print("init embeddings")
        
        # pdb.set_trace()
        self.embeds = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
        print("init emebdding finished")

        print("init first order part")
        self.first_order_weight = nn.ModuleList([nn.ParameterList([nn.Parameter(torch.randn(self.embedding_size), requires_grad=True) for j in range(self.field_size)]) for i in range(self.deep_layers[0])])
        self.bias = nn.Parameter(torch.randn(self.deep_layers[0]), requires_grad=True)
        print("init first order part finished")

        print("init second order part")
        self.inner_second_weight_emb = nn.ModuleList([nn.ParameterList([nn.Parameter(torch.randn(self.embedding_size), requires_grad=True) for j in range(self.field_size)]) for i in range(self.deep_layers[0])])
        print("init second order part finished")

        print("init nn part")
        for i, h in enumerate(self.deep_layers[1:], 1):
            setattr(self, 'linear_' + str(i), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
            setattr(self, 'linear_' + str(i) + '_dropout', nn.Dropout(self.dropout_deep[i]))
        self.deep_last_layer = nn.Linear(self.deep_layers[-1], self.n_class)
        print("init nn part succeed")

        print("init succeed")

    def forward(self, Xi, Xv):
        # self.X = [X for i in range(self.num_inputs)]
        # self.y = y
        # self.keep_prob_train = 1 - np.array(self.drop_out)
        # self.keep_prob_test = np.ones_like(self.drop_out)
        # self.layer_keeps = layer_keeps
        # self.vars = utils.init_var_map(self.init_vars, self.init_path)
        # w0 = [self.vars['embed_%d' % i] for i in range(self.num_inputs)]
        # Xw = torch.cat([torch.mm(self.X[i], w0[i]) for i in range(num_inputs)], 1)
        # xw3d = torch.reshape(Xw, (-1, self.num_inputs, self.embed_size))
        
        # row = []
        # col = []
        # for i in range(self.num_inputs - 1):
        #     for j in range(i + 1, self.num_inputs):
        #         row.append(i)
        #         col.append(j)
            
        # p = torch.transpose(torch.gather(torch.transpose(xw3d, (1, 0, 2)), row), ())

        # Xi: indexs, b x k x 1
        # Xv: value, b x k
        # emb_arr: b x embs
        # [8, 25, 445852, 36, 371, 4, 11328, 33995, 12, 7, 5, 4, 20, 2, 38, 6]
        # pdb.set_trace()
        emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t()*Xv[:, i]).t() for i, emb in enumerate(self.embeds)]
        first_order_arr = []
        for i, weight_arr in enumerate(self.first_order_weight):
            tmp_arr = []
            for j, weight in enumerate(weight_arr):
                tmp_arr.append(torch.sum(emb_arr[j] * weight, 1))
            first_order_arr.append(sum(tmp_arr).view([-1, 1]))
        first_order = torch.cat(first_order_arr, 1)

        inner_product_arr = []
        for i, weight_arr in enumerate(self.inner_second_weight_emb):
            tmp_arr = []
            for j, weight in enumerate(weight_arr):
                tmp_arr.append(torch.sum(emb_arr[j] * weight, 1))
            sum_ = sum(tmp_arr)
            inner_product_arr.append((sum_ * sum_).view([-1, 1]))
        inner_product = torch.cat(inner_product_arr, 1)
        first_order = first_order + inner_product

        if self.deep_layers_activation == 'sigmoid':
            activation = torch.sigmoid
        elif self.deep_layers_activation == 'tanh':
            activation = torch.tanh
        else:
            activation = torch.relu 
        x_deep = first_order
        for i, h in enumerate(self.deep_layers[1: ], 1):
            x_deep = getattr(self, 'linear_' + str(i))(x_deep)
            x_deep = activation(x_deep)
            x_deep = getattr(self, 'linear_' + str(i) + '_dropout')(x_deep)
        x_deep = self.deep_last_layer(x_deep)
        return torch.sum(x_deep, 1)
    
    def fit(self, data, Xi_valid=None, Xv_valid=None, y_vaild=None, ealry_stopping=False, refit=False, save_path=None):
        print("pre_process data ing...")
        is_valid = False
        # Xi_train = np.array(Xi_train).reshape((-1, self.field_size, 1))
        # Xv_train = np.array(Xv_train)
        # y_train = np.array(y_train)
        # x_size = Xi_train.shape[0]
        if Xi_valid:
            Xi_valid = np.array(Xi_valid).reshape((-1, self.field_size, 1))
            Xv_valid = np.array(Xv_valid)
            y_vaild = np.array(y_vaild)
            x_valid_size = Xi_valid.shape[0]
            is_valid = True
        print("pre_process data finished")

        model = self.train()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "rmsp":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "adag":
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        
        criterion = F.binary_cross_entropy_with_logits

        train_result = []
        valid_result = []
        # pdb.set_trace()
        data_loader = DataLoader(data, batch_size=512*64, shuffle=True, num_workers=0)
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            # batch_iter = x_size // self.batch_size
            epoch_begin_time = time()
            batch_begin_time = time()
            print("start epoch: ", epoch, "time: ", epoch_begin_time)
            for step, (batch_y, batch_xi, batch_xv) in enumerate(data_loader):
                # offset = i * self.batch_size
                # end = min(x_size, offset + self.batch_size)
                # if offset == end:
                #     break
                # pdb.set_trace()
                # batch_xi = torch.tensor(torch.LongTensor(batch_xi))
                # batch_xv = torch.tensor(torch.FloatTensor(batch_xv))
                # batch_y = torch.tensor(torch.FloatTensor(batch_y))
                batch_xi = batch_xi.reshape((-1, self.field_size + 1, 1))
                optimizer.zero_grad()
                outputs = model(batch_xi, batch_xv)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                # pdb.set_trace()
                total_loss += loss.item()
                if step % 100 == 99:
                    evalb = self.evaluate(batch_xi, batch_xv, batch_y)
                    print("[%d, %5d] loss: %.6f metric: %.6f time: %.1f s" % (epoch + 1, step + 1, total_loss, evalb, time() - batch_begin_time))
                    total_loss = 0.0
                    batch_begin_time = time()
            # 
            train_loss, trian_eval = self.eval_by_batch(data)
            train_result.append(trian_eval)
            print('*' * 50)
            print('[%d] loss: %.6f metric: %.6f time: %.1f s' % (epoch + 1, train_loss, trian_eval, time() - epoch_begin_time))
            print('*' * 50)

            if is_valid:
                valid_loss, valid_eval = self.eval_by_batch(Xi_valid, Xv_valid, y_vaild, x_valid_size)
                valid_result.append(valid_eval)
                print('*' * 50)
                print('[%d] loss: %.6f metric: %.6f time: %.1f s' % (epoch + 1, valid_loss, valid_eval, time() - epoch_begin_time))
                print('*' * 50)
            if save_path:
                torch.save(self.state_dict(), save_path)
            if is_valid and ealry_stopping and self.training_termination(valid_result):
                print("early stop at [%d] epoch!" % (epoch + 1))
                break
        
        if is_valid and refit:
            print("refitting the model")
            if self.greater_is_better:
                best_epoch = np.argmax(valid_result)
            else:
                best_epoch = np.argmin(valid_result)
            best_train_score = train_result[best_epoch]
            Xi_train = np.concatenate((Xi_train, Xi_valid))
            Xv_train = np.concatenate((Xv_train, Xi_valid))
            y_train = np.concatenate((y_train, y_vaild))
            x_size = x_size + x_valid_size
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            for epoch in range(64):
                batch_iter = x_size // self.batch_size
                for i in range(batch_iter + 1):
                    offset = i * self.batch_size
                    end = min(x_size, offset + self.batch_size)
                    if offset == end:
                        break
                    batch_xi = torch.tensor(torch.LongTensor(Xi_train[offset: end]))
                    batch_xv = torch.tensor(torch.FloatTensor(Xv_train[offset: end]))
                    batch_y = torch.tensor(torch.FloatTensor(y_train[offset: end]))
                    optimizer.zero_grad()
                    outputs - model(batch_xi, batch_xv)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                trian_loss, train_eval = self.eval_by_batch(Xi_train, Xv_train, y_train, x_size)
                if save_path:
                    torch.save(self.state_dict().save_path)
                if abs(best_train_score - train_eval) < 0.001 or (self.greater_is_better and train_eval > best_train_score) or ((not self.greater_is_better) and train_result < best_train_score):
                    break
            print("refit finished")
            

    def eval_by_batch(self, data):
        total_loss = 0.0
        y_pred = []
        y_true = []
        data_loader = DataLoader(data, 512*128, num_workers=0)
        # batch_iter = x_size // batch_size
        criterion = F.binary_cross_entropy_with_logits
        model = self.eval()
        # for i in range(batch_iter + 1):
        for step, (batch_y, batch_xi, batch_xv) in enumerate(data_loader):
            # offset = i * batch_size
            # end = min(x_size, offset + batch_size)
            # if offset == end:
                # break
            # batch_xi = torch.tensor(torch.LongTensor(Xi[offset: end]))
            # batch_xv = torch.tensor(torch.FloatTensor(Xv[offset: end]))
            # batch_y = torch.tensor(torch.FloatTensor(y[offset: end]))
            batch_xi = batch_xi.reshape((-1, self.field_size + 1, 1))
            outputs = model(batch_xi, batch_xv)
            pred = F.sigmoid(outputs).cpu()
            y_pred.extend(pred.data.numpy())
            y_true.extend(batch_y.data.numpy())
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * len(batch_y)
        total_metric = self.eval_metric(y_true, y_pred)
        return total_loss / len(data), total_metric
    
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
    
    def training_termination(self, valid_result):
        # todo fix
        if len(valid_result) > 4:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                    valid_result[-2] < valid_result[-3] and \
                        valid_result[-3] < valid_result[-4]:
                        return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                    valid_result[-2] > valid_result[-1] and \
                        valid_result[-3] > valid_result[-4]:
                        return True
        return False
    
    def predict(self, Xi, Xv):
        Xi = np.array(Xi).reshape((-1, self.field_size, 1))
        Xi = torch.tensor(torch.LongTensor(Xi))
        Xv = torch.tensor(torch.FloatTensor(Xv))
        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv))
        return (pred.data.numpy() > 0.5)
    
    def predict_proba(self, Xi, Xv):
        Xi = np.array(Xi).reshape((-1, self.field_size, 1))
        Xi = torch.tensor(torch.LongTensor(Xi))
        Xv = torch.tensor(torch.FloatTensor(Xv))
        
        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv))
        return pred.data.numpy()
    
    def inner_predict(self, Xi, Xv):
        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv))
        return (pred.data.numpy() > 0.5)
    
    def inner_predict_proba(self, Xi, Xv):
        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv))
        return pred.data.numpy()
    
    def evaluate(self, Xi, Xv, y):
        y_pred = self.inner_predict_proba(Xi, Xv)
        return self.eval_metric(y.data.numpy(), y_pred)
    
