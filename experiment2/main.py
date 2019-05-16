import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from IPython.core.debugger import Pdb

from dataset import VQADataset, VQABatchSampler
from train import train_model
from vqa import VQAModel
from scheduler import CustomReduceLROnPlateau

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config.yml')

def load_datasets(config, phases):
    datasets = {x: VQADataset(mode=x, preprocess=config['data']['preprocess']) for x in phases}
    batch_samplers = {x: VQABatchSampler(datasets[x], config['data']['batch_size']) for x in phases}
    num_workers = config['data']['num_workers']
    dataloaders = {x: DataLoader(datasets[x], batch_sampler=batch_samplers[x], num_workers=num_workers) for x in phases}
    print("dataset size", {x: len(datasets[x]) for x in phases})
    print("ques vocab size: {}".format(len(VQADataset.ques_vocab)))
    print("ans vocab size: {}".format(len(VQADataset.ans_vocab)))
    return dataloaders, VQADataset.ques_vocab, VQADataset.ans_vocab

def main(config):
    phases = ['train', 'val']
    dataloaders, ques_vocab, ans_vocab = load_datasets(config, phases)

    config['model']['params']['vocab_size'] = len(ques_vocab)
    config['model']['params']['output_size'] = len(ans_vocab) #sasaki # originally len(ans_vocab)-1 

    model = VQAModel(**config['model']['params']) 
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), **config['optim']['params']) 
    # the original code has 'class' (SGD/RMSprop/Adam) options

    best_acc = 0 #sasaki
    startEpoch = 0
    if 'reload' in config['model']:
        pathForTrainedModel = os.path.join(config['save_dir'], config['model']['reload'])
        if os.path.exists(pathForTrainedModel):
            print("=> loading checkpoint/model found at '{0}'".format(pathForTrainedModel))
            checkpoint = torch.load(pathForTrainedModel)
            startEpoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
    if config['use_gpu']:
        model = model.cuda()

    save_dir = os.path.join(os.getcwd(), config['save_dir'])

    if 'scheduler' in config['optim'] and config['optim']['scheduler'].lower() == 'CustomReduceLROnPlateau'.lower():
        print('CustomReduceLROnPlateau')
        exp_lr_scheduler = CustomReduceLROnPlateau(
            optimizer, config['optim']['scheduler_params']['maxPatienceToStopTraining'], config['optim']['scheduler_params']['base_class_params'])
    else:
        # Decay LR by a factor of gamma every step_size epochs
        print('lr_scheduler.StepLR')
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print("begin training")
    model = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, save_dir,
                        num_epochs=config['optim']['n_epochs'], use_gpu=config['use_gpu'], best_accuracy=best_acc, start_epoch=startEpoch)

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    args.config = os.path.join(os.getcwd(), args.config)
    config = yaml.load(open(args.config))
    config['use_gpu'] = config['use_gpu'] and torch.cuda.is_available()

    main(config)
