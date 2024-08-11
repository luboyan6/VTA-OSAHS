import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from utils.MLP_model import MLP
from data_processing import read_dataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from utils.ordinal_loss import OrdinalRegressionLoss

def load_data():
    data_path = r'yourpath'
    dataset = read_dataset(data_path)

    features = []
    labels = []

    for data in dataset:
        feature = data.view(769)[0:768].tolist()
        label = int(data.view(769)[-1])
        features.append(feature)
        labels.append(label)

    return torch.tensor(features), torch.tensor(labels)

def train():

    torch.manual_seed(42)

    input_size = 768
    hidden_size = 256
    output_size = 4
    lr = 0.001
    batch_size = 4
    test_batch_size = 4
    num_epochs = 100
    dropout = 0.4

    X,y = load_data()

    ros = RandomOverSampler()
    ros.set_params(shrinkage=0.1)

    X_resampled, y_resampled = ros.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    model = MLP(input_size, hidden_size, output_size, dropout)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    ordinal_loss = OrdinalRegressionLoss(num_class=batch_size,train_cutpoints=True,scale = 4.0)

    best_acc = 0.0

    # train
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            max_probs, _ = torch.max(outputs, dim=1)
            max_probs = max_probs.unsqueeze(-1)
            labels = labels.unsqueeze(-1)
            loss, likelihoods = ordinal_loss(max_probs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            y_pred = torch.argmax(likelihoods, dim=1)
            running_acc += accuracy_score(labels, y_pred)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                max_probs, _ = torch.max(outputs, dim=1)
                max_probs = max_probs.unsqueeze(-1)
                labels_ = labels.unsqueeze(-1)
                loss, likelihoods = ordinal_loss(max_probs, labels_)
                val_loss += loss.item()
                y_true = labels_.tolist()
                y_pred = torch.argmax(likelihoods, dim=1).tolist()
                val_acc += accuracy_score(y_true, y_pred)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}')

        # Test
        model.eval()
        test_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                max_probs, _ = torch.max(outputs, dim=1)
                max_probs = max_probs.unsqueeze(-1)
                labels_ = labels.unsqueeze(-1)

                loss, likelihoods = ordinal_loss(max_probs, labels_)
                preds_index = torch.argmax(likelihoods, dim=1)
                test_loss += loss.item()

                y_true.extend(labels_.tolist())
                y_pred.extend(preds_index.tolist())

        test_acc = accuracy_score(y_true, y_pred)
        test_loss = test_loss / len(test_loader)

        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

        if test_acc > best_acc:
            best_acc = test_acc
            best_state_dict = model.state_dict()
            torch.save(best_state_dict, r'yourpath')

if __name__ == '__main__':
    train()
