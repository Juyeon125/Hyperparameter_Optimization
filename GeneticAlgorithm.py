import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
import time
from datetime import datetime
from deap import base, creator, tools, algorithms

# Hyperparameters
class Config:
    AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
    MAX_SEQ_LENGTH = 1000
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    EPOCHS = 30
    MOMENTUM = 0.99
    EPSILON = 1e-02
    DROPOUT_PROB = 0.5
    TEST_SIZE = 0.2
    VAL_SIZE = 0.16
    RANDOM_STATE = 20

def one_hot_encoding(data):
    matrix = np.zeros((Config.MAX_SEQ_LENGTH, len(Config.AMINO_ACIDS)), dtype=np.float64)
    count = 0
    for i in data:
        if count >= Config.MAX_SEQ_LENGTH:
            break
        else:
            matrix[count][Config.AMINO_ACIDS.index(i)] = 1
            count += 1
    return matrix

class ECDataset:
    def __init__(self, seq, label):
        self.seq = seq
        self.label = label

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq = one_hot_encoding(self.seq[idx])
        seq = np.array([seq])
        seq = torch.FloatTensor(seq)
        label = np.array([EC_LABEL.index(self.label[idx])])
        label = torch.LongTensor(label)
        return seq, label

class ECModel(nn.Module):
    def __init__(self, dropout_prob):
        super(ECModel, self).__init__()

        conv1 = nn.Conv2d(1, 128, kernel_size=(4, 21), stride=1, dilation=1)
        pool1 = nn.MaxPool2d(kernel_size=(997, 1))

        conv2 = nn.Conv2d(1, 128, kernel_size=(8, 21), stride=1, dilation=1)
        pool2 = nn.MaxPool2d(kernel_size=(993, 1))

        conv3 = nn.Conv2d(1, 128, kernel_size=(16, 21), stride=1, dilation=1)
        pool3 = nn.MaxPool2d(kernel_size=(985, 1))

        batch1 = nn.BatchNorm2d(128, momentum=Config.MOMENTUM, eps=Config.EPSILON)
        
        num_label = len(EC_LABEL)
        
        fc1 = nn.Linear(384, 512)
        fc2 = nn.Linear(512, 512)
        fc3 = nn.Linear(512, num_label)
        
        batch_fc1 = nn.BatchNorm1d(512, momentum=Config.MOMENTUM, eps=Config.EPSILON)
        batch_fc2 = nn.BatchNorm1d(512, momentum=Config.MOMENTUM, eps=Config.EPSILON)
        batch_fc3 = nn.BatchNorm1d(num_label, momentum=Config.MOMENTUM, eps=Config.EPSILON)

        self.init_layer(conv1)
        self.init_layer(conv2)
        self.init_layer(conv3)
        self.init_layer(fc1)
        self.init_layer(fc2)
        self.init_layer(fc3)
        
        dropout = nn.Dropout(p=dropout_prob)
        self.softmax = nn.Softmax(dim=1)
        self.conv1_module = nn.Sequential(conv1, batch1, nn.ReLU(), pool1, nn.Flatten())
        self.conv2_module = nn.Sequential(conv2, batch1, nn.ReLU(), pool2, nn.Flatten())
        self.conv3_module = nn.Sequential(conv3, batch1, nn.ReLU(), pool3, nn.Flatten())
        
        self.fc_module = nn.Sequential(fc1, batch_fc1, nn.ReLU(), 
                                       fc2, batch_fc2, nn.ReLU(), 
                                       fc3, batch_fc3)

    def forward(self, x):
        out1 = self.conv1_module(x)
        out2 = self.conv2_module(x)
        out3 = self.conv3_module(x)
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.fc_module(out)
        return out

    def init_layer(self, layer):
        nn.init.xavier_normal_(layer.weight)
        layer.bias.data.fill_(0)
        return layer

class SklearnECModel(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=1e-4, dropout_prob=0.5, momentum=0.99, epsilon=1e-2, batch_size=64, epochs=30):
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.momentum = momentum
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = ECModel(dropout_prob).cuda() 
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.cross_loss = nn.CrossEntropyLoss(weight=torch.Tensor(class_weight).cuda()) 

    def fit(self, X, y):
        print(f"Starting training with params: learning_rate={self.learning_rate}, dropout_prob={self.dropout_prob}, momentum={self.momentum}, epsilon={self.epsilon}, batch_size={self.batch_size}, epochs={self.epochs}")
        train_dataloader = DataLoader(traindataset, batch_size=self.batch_size, pin_memory=True, num_workers=1, drop_last=True)
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for data in train_dataloader:
                seq, label = data
                seq = seq.cuda()
                label = label.view(-1).cuda()
                
                self.optimizer.zero_grad()
                output = self.model(seq)
                loss = self.cross_loss(output, label)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_dataloader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}")

        return self

    def predict(self, X):
        self.model.eval()
        dataloader = DataLoader(X, batch_size=self.batch_size, pin_memory=True, num_workers=1, drop_last=False)
        all_preds = []
        with torch.no_grad():
            for data in dataloader:
                seq, _ = data
                seq = seq.cuda()
                output = self.model(seq)
                preds = output.argmax(dim=1, keepdim=True)
                all_preds.extend(preds.cpu().numpy())
        return np.array(all_preds).flatten()

    def score(self, X, y):
        y = np.array([EC_LABEL.index(label) for label in y]) 
        val_dataloader = DataLoader(X, batch_size=self.batch_size, pin_memory=True, num_workers=1, drop_last=True)
        correct = 0
        all_labels = []
        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for data in val_dataloader:
                seq, label = data
                seq = seq.cuda() 
                label = label.view(-1).cuda() 
                output = self.model(seq)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
        
        accuracy = correct / len(val_dataloader.dataset)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Validation Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        return f1

def create_logger():
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        formatter = logging.Formatter('[%(asctime)s] - %(message)s')
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)
        logger.setLevel(level=logging.DEBUG)
    return logger

def evaluate(params):
    learning_rate, dropout_prob, momentum, epsilon, batch_size, epochs = params
    model = SklearnECModel(learning_rate=learning_rate, dropout_prob=dropout_prob, momentum=momentum, epsilon=epsilon, batch_size=batch_size, epochs=epochs)
    cv = StratifiedKFold(n_splits=3)
    f1_scores = []
    for train_idx, val_idx in cv.split(traindataset, train_label):
        train_data = DataLoader([traindataset[i] for i in train_idx], batch_size=batch_size, pin_memory=True, num_workers=1, drop_last=True)
        val_data = DataLoader([traindataset[i] for i in val_idx], batch_size=batch_size, pin_memory=True, num_workers=1, drop_last=True)
        model.fit(train_data, [train_label[i] for i in train_idx])
        f1_scores.append(model.score(val_data, [train_label[i] for i in val_idx]))
    return np.mean(f1_scores),

def main():
    dataset_df = pd.read_csv("uniprot_swiss_enzymes.csv")
    seq_data = list(dataset_df["SEQ"])
    label_data = list(dataset_df["EC"])

    df = pd.DataFrame({"SEQ": seq_data, "EC": label_data})
    df = df.groupby("EC").filter(lambda x: len(x) >= 100)
    seq_data = list(df["SEQ"])
    label_data = list(df["EC"])
    
    global EC_LABEL
    EC_LABEL = sorted(list(set(label_data)))
    print(f"Filtered dataset to {len(seq_data)} samples with {len(EC_LABEL)} unique classes.")

    train_seq, val_seq, train_label, val_label = train_test_split(seq_data, label_data, random_state=Config.RANDOM_STATE, shuffle=True, test_size=Config.VAL_SIZE)
    val_seq, test_seq, val_label, test_label = train_test_split(val_seq, val_label, random_state=Config.RANDOM_STATE, shuffle=True, test_size=Config.TEST_SIZE)
    
    global traindataset, valdataset, testdataset, class_weight
    traindataset = ECDataset(train_seq, train_label)
    valdataset = ECDataset(val_seq, val_label)
    testdataset = ECDataset(test_seq, test_label)
    
    class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(label_data), y=label_data)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.rand)
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_float, toolbox.attr_float, toolbox.attr_float, toolbox.attr_float, toolbox.attr_float, toolbox.attr_float), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
    
    population = toolbox.population(n=20)
    ngen = 10
    cxpb = 0.5
    mutpb = 0.2
    
    start_time = time.time()
    print(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    
    for gen in range(ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        population = toolbox.select(offspring, k=len(population))
        fits = [ind.fitness.values[0] for ind in population]
        print(f"Generation {gen}, Best F1 Score: {max(fits)}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Genetic Algorithm optimization complete. Time taken: {elapsed_time:.2f} seconds.")

    best_ind = tools.selBest(population, k=1)[0]
    best_params = {
        'learning_rate': best_ind[0] * (1e-1 - 1e-5) + 1e-5,
        'dropout_prob': best_ind[1] * (0.5 - 0.1) + 0.1,
        'momentum': best_ind[2] * (0.99 - 0.8) + 0.8,
        'epsilon': best_ind[3] * (1e-2 - 1e-8) + 1e-8,
        'batch_size': int(best_ind[4] * (128 - 32) + 32),
        'epochs': int(best_ind[5] * (30 - 30) + 30)
    }
    
    print(f"Best Parameters: {best_params}")
    
    model = SklearnECModel(**best_params)
    model.fit(traindataset, train_label)

    train_dataloader = DataLoader(traindataset, batch_size=best_params['batch_size'], pin_memory=True, num_workers=1, drop_last=True)
    val_dataloader = DataLoader(valdataset, batch_size=best_params['batch_size'], pin_memory=True, num_workers=1, drop_last=True)
    test_dataloader = DataLoader(testdataset, batch_size=best_params['batch_size'], pin_memory=True, num_workers=1, drop_last=True)

    logger = create_logger()

    for epoch in range(best_params['epochs']):
        train_loss = 0
        cur_val_loss = 0

        for tr_i, data in enumerate(train_dataloader):
            model.model.train()
            torch.cuda.empty_cache()

            seq, label = data
            seq = seq.cuda()
            label = label.view(-1).cuda()

            model.optimizer.zero_grad()
            output = model.model(seq)
            loss = model.cross_loss(output, label)
            loss.backward()
            model.optimizer.step()

            train_loss += loss.item()

        train_loss /= (len(train_dataloader) * best_params['batch_size'])

        with torch.no_grad():
            for va_i, data in enumerate(val_dataloader):
                model.model.eval()
                torch.cuda.empty_cache()
                seq, label = data
                seq = seq.cuda()
                label = label.view(-1).cuda()
                output = model.model(seq)
                loss = model.cross_loss(output, label)
                cur_val_loss += loss.item()

            cur_val_loss /= (len(val_dataloader) * best_params['batch_size'])
            acc_count = 0

            test_pred = []
            test_true = []
            test_output = []

            for te_i, data in enumerate(test_dataloader):
                seq, label = data
                seq = seq.cuda()
                label = label.view(-1).cuda()
                output = model.model(seq)

                pred = output.argmax(dim=1, keepdim=True)
                true = label.view_as(pred)

                test_pred.extend(pred.cpu().numpy())
                test_true.extend(true.cpu().numpy())
                test_output.extend(F.softmax(output, dim=1).cpu().numpy())

                acc_count += (pred == true).sum().item()

            acc = acc_count / (len(test_dataloader) * best_params['batch_size']) * 100

            test_pred = np.array(test_pred).flatten()
            test_true = np.array(test_true).flatten()
            test_output = np.array(test_output)

            accuracy = accuracy_score(test_true, test_pred)
            precision = precision_score(test_true, test_pred, average='weighted', zero_division=0)
            recall = recall_score(test_true, test_pred, average='weighted', zero_division=0)
            f1 = f1_score(test_true, test_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(test_true, test_pred)

            n_classes = test_output.shape[1]

            y_true_one_hot = np.zeros((test_true.size, n_classes))
            y_true_one_hot[np.arange(test_true.size), test_true] = 1

            unique_classes_in_y_true = np.unique(test_true)
            if len(unique_classes_in_y_true) > 1 and n_classes == len(unique_classes_in_y_true):
                auc = roc_auc_score(y_true_one_hot, test_output, multi_class='ovr', average='weighted')
            else:
                auc = float('nan')

            logger.info(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {cur_val_loss:.4f}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}')

if __name__ == "__main__":
    main()
