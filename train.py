import os
import sys
import time
import re
import argparse
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.models.dpc_rnn import DPC_RNN
from src.trainer import Trainer
from src.data.get_dataiter import get_data
from config.options import set_args
from utils import AverageMeter, denorm
import torchvision.utils as vutils

plt.switch_backend('agg')

torch.manual_seed(0)
np.random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    args = set_args()
    train_loader = get_data(args, 'train')
    val_loader = get_data(args, 'val')

    if args.model == 'dpc-rnn':
        model = DPC_RNN(sample_size=args.img_dim,
                        num_seq=args.num_seq,
                        seq_len=args.seq_len,
                        network=args.net,
                        pred_step=args.pred_step)
    else:
        raise ValueError('wrong model!')

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(args, model, criterion)

    ### restart training ###
    start_epoch, best_acc, iteration = 0, 0, 0

    if args.resume:
        start_epoch, iteration, best_acc = trainer.resume(args)

    if args.pretrain:
        trainer.pretrain(args)

    # setup tools
    args.old_lr = None
    img_path, model_path = trainer.set_path(args)
    # print(img_path, model_path)
    # exit()
    try:  # old version
        writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    except:  # v1.7
        writer_val = SummaryWriter(logdir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc, train_accuracy_list = train(args, train_loader, trainer, iteration, epoch, writer_train)
        val_loss, val_acc, val_accuracy_list = validate(args, val_loader, trainer, epoch)

        # save curve
        writer_train.add_scalar('global/loss', train_loss, epoch)
        writer_train.add_scalar('global/accuracy', train_acc, epoch)
        writer_val.add_scalar('global/loss', val_loss, epoch)
        writer_val.add_scalar('global/accuracy', val_acc, epoch)
        writer_train.add_scalar('accuracy/top1', train_accuracy_list[0], epoch)
        writer_train.add_scalar('accuracy/top3', train_accuracy_list[1], epoch)
        writer_train.add_scalar('accuracy/top5', train_accuracy_list[2], epoch)
        writer_val.add_scalar('accuracy/top1', val_accuracy_list[0], epoch)
        writer_val.add_scalar('accuracy/top3', val_accuracy_list[1], epoch)
        writer_val.add_scalar('accuracy/top5', val_accuracy_list[2], epoch)

        # save check_point
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        trainer.save_checkpoint(epoch, best_acc, iteration, is_best, filename=os.path.join(
            model_path, 'epoch%s.pth.tar' % str(epoch + 1)), keep_all=False)

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))


def train(args, data_loader, trainer, iteration, epoch, writer_train):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    for idx, input_seq in enumerate(data_loader):
        trainer.train_step(args, idx, input_seq, iteration, epoch, writer_train, losses, accuracy, accuracy_list)
    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]

def validate(args, data_loader, trainer, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]

    with torch.no_grad():
        for idx, input_seq in tqdm(enumerate(data_loader), total=len(data_loader)):
            trainer.valid_step(idx, input_seq, losses, accuracy, accuracy_list)

    print('[{0}/{1}] Loss {loss.local_avg:.4f}\t'
          'Acc: top1 {2:.4f}; top3 {3:.4f}; top5 {4:.4f} \t'.format(
        epoch, args.epochs, *[i.avg for i in accuracy_list], loss=losses))
    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]


if __name__ == "__main__":
    main()
