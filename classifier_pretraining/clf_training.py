import os
import sys
import torch
import tqdm
import argparse

import numpy as np
import torchvision.models as models
from sklearn.metrics import confusion_matrix

# set directory to parent directory, one level up
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# print current working directory
print("Current Working Directory:", os.getcwd())

from utils.data_utils import get_data, get_gen

dataset = 'cubicc'


# Configurations
configs = {
    "data": {
        "data_name": dataset,
        "num_clusters_data": 10,
    },
    "training": {
        "batch_size": 32,
        "augment": False,
        "augmentation_method": 'simple',
        "aug_decisions_weight": 1,
    },
    "globals": {
        "seed": 42,
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, help='the dataset')

    args = parser.parse_args()
    dataset = args.data_name
    configs['data']['data_name'] = args.data_name

    # Set seed
    torch.manual_seed(configs['globals']['seed'])
    np.random.seed(configs['globals']['seed'])

    # Get data
    trainset, trainset_eval, testset = get_data(configs)

    gen_train = get_gen(trainset, configs, validation=False, shuffle=False)
    gen_train_eval = get_gen(trainset_eval, configs, validation=True, shuffle=False)
    gen_test = get_gen(testset, configs, validation=True, shuffle=False)

    y_train = trainset_eval.dataset.targets.clone().detach()[trainset_eval.indices].numpy()
    y_test = testset.dataset.targets.clone().detach()[testset.indices].numpy()

    #print("\nTrainset shape:", trainset.dataset.data.shape)
    print("Number of samples in trainset:", len(gen_train.dataset))

    #print("\nTrainset eval shape:", trainset_eval.dataset.data.shape)
    print("Number of samples in trainset eval:", len(gen_train_eval.dataset))

    #print("\nTestset shape:", testset.dataset.data.shape)
    print("Number of samples in testset:", len(gen_test.dataset))

    print(50 * "-")

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Load model 
    if not os.path.exists('classifier_pretraining/resnet50.pth'):
        model = models.resnet50(pretrained=True)
        torch.save(model, 'classifier_pretraining/resnet50.pth')
    else:
        model = torch.load('classifier_pretraining/resnet50.pth')
    
    # adjust first layer for greyscale images
    if dataset == 'mnist' or dataset == 'fmnist':
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    model = model.to(device)

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Epochs
    n_epochs = 50

    # Training
    model.train()
    for epoch in tqdm.tqdm(range(n_epochs)):
        for i, (x_batch, y_batch) in enumerate(gen_train):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(x_batch)

            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            # use gen_train_eval to evaluate the model
            if i % 10 == 0:
                model.eval()
                with torch.no_grad():
                    loss_eval = []
                    for x_eval, y_eval in gen_train_eval:
                        x_eval, y_eval = x_eval.to(device), y_eval.to(device)
                        y_pred_eval = model(x_eval)
                        loss_eval.append(criterion(y_pred_eval, y_eval))

                loss_eval = torch.mean(torch.stack(loss_eval))
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}, Loss Eval: {loss_eval.item()}")
                model.train()

    # Save model
    torch.save(model.state_dict(), f"classifier_pretraining/resnet50_{configs['data']['data_name']}.pth")

    # Evaluation
    model.eval()

    with torch.no_grad():
        for i, (x, y) in enumerate(gen_test):
            x, y = x.to(device), y.to(device)
            if i == 0:
                y_pred = model(x)
            else:
                y_pred = torch.cat([y_pred, model(x)], dim=0)

    y_pred = y_pred.argmax(dim=1).cpu().numpy()

    print("Accuracy:", np.mean(y_pred == y_test))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()




