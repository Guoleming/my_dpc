import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.resnet_2d3d import neq_load_customized
import torch.optim as optim
from tensorboardX import SummaryWriter
from utils import denorm, calc_topk_accuracy
import torchvision.utils as vutils
import logging
import utils
import os
import re
import glob
import time


class Trainer(object):
    def __init__(self, args, model, criterion):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.model = self.model.to("cuda")
            self.criterion = self.criterion.to("cuda")

        # optimizer
        if args.train_what == 'last':  # fine-tune????
            for name, param in model.module.resnet.named_parameters():
                param.requires_grad = False
        else:
            pass  # train all layers

        # print('\n===========Check Grad============')
        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)
        # print('=================================\n')

        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

        print('| num. module params: {} (num. trained: {})'.format(
            sum(p.numel() for p in params),
            sum(p.numel() for p in params if p.requires_grad),
        ))

        self.de_normalize = denorm()

    def resume(self, args):
        if os.path.isfile(args.resume):
            args.old_lr = float(re.search('_lr(.+?)_', args.resume).group(1))
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            self.model.load_state_dict(checkpoint['state_dict'])
            if not args.reset_lr:  # if didn't reset lr, load old optimizer
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('==== Change lr from %f to %f ====' % (args.old_lr, args.lr))
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.resume))
            exit()
        return start_epoch, iteration, best_acc

    def pretrain(self, args):
        if os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            self.model = neq_load_customized(self.model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})"
                  .format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    def train_step(self, args, idx, input_seq, iteration, epoch, writer_train, losses, accuracy, accuracy_list):
        self.model.train()
        self.criterion.train()
        self.optimizer.zero_grad()

        tic = time.time()
        input_seq = self._prepare_sample(input_seq)
        B = input_seq.size(0)
        [score_, mask_] = self.model(input_seq)

        # visualize
        if (iteration == 0) or (iteration == args.print_freq):
            if B > 2: input_seq = input_seq[0:2, :]
            writer_train.add_image('input_seq',
                                   self.de_normalize(vutils.make_grid(
                                       input_seq.transpose(2, 3).contiguous().view(-1, 3, args.img_dim, args.img_dim),
                                       nrow=args.num_seq * args.seq_len)),
                                   iteration)
        del input_seq

        if idx == 0:
            self.target_, (_, self.B2, self.NS, self.NP, self.SQ) = process_output(mask_)

        score_flattened = score_.view(B * self.NP * self.SQ, self.B2 * self.NS * self.SQ)
        target_flattened = self.target_.view(B * self.NP * self.SQ, self.B2 * self.NS * self.SQ)
        target_flattened = target_flattened.long().argmax(dim=1)

        loss = self.criterion(score_flattened, target_flattened)
        top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1, 3, 5))

        accuracy_list[0].update(top1.item(), B)
        accuracy_list[1].update(top3.item(), B)
        accuracy_list[2].update(top5.item(), B)

        losses.update(loss.item(), B)
        accuracy.update(top1.item(), B)

        del score_

        loss.backward()
        self.optimizer.step()

        del loss

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}]\t'
                  'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                  'Acc: top1 {2:.4f}; top3 {3:.4f}; top5 {4:.4f} T:{5:.2f}\t'.format(
                epoch, idx, top1, top3, top5, time.time() - tic, loss=losses))

            writer_train.add_scalar('local/loss', losses.val, iteration)
            writer_train.add_scalar('local/accuracy', accuracy.val, iteration)

            iteration += 1
        # return losses, accuracy, accuracy_list

    def valid_step(self, idx, input_seq, losses, accuracy, accuracy_list):
        self.model.eval()
        self.criterion.eval()

        input_seq = self._prepare_sample(input_seq)
        B = input_seq.size(0)
        [score_, mask_] = self.model(input_seq)
        del input_seq

        if idx == 0:
            target_, (_, self.B2, NS, NP, SQ) = process_output(mask_)

        # [B, P, SQ, B, N, SQ]
        score_flattened = score_.view(B * NP * SQ, B2 * NS * SQ)
        target_flattened = target_.view(B * NP * SQ, B2 * NS * SQ)
        target_flattened = target_flattened.argmax(dim=1)

        loss = self.criterion(score_flattened, target_flattened)
        top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1, 3, 5))

        losses.update(loss.item(), B)
        accuracy.update(top1.item(), B)

        accuracy_list[0].update(top1.item(), B)
        accuracy_list[1].update(top3.item(), B)
        accuracy_list[2].update(top5.item(), B)
        # return losses, accuracy, accuracy_list

    def set_path(self, args):
        if args.resume:
            exp_path = os.path.dirname(os.path.dirname(args.resume))
        else:
            exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_' \
                       'bs{args.batch_size}_lr{1}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_' \
                       'train-{args.train_what}{2}'.format(
                'r%s' % args.net[6::], \
                args.old_lr if args.old_lr is not None else args.lr, \
                '_pt=%s' % args.pretrain.replace('/', '-') if args.pretrain else '', \
                args=args)
        img_path = os.path.join(exp_path, 'img')
        model_path = os.path.join(exp_path, 'model')
        if not os.path.exists(img_path): os.makedirs(img_path)
        if not os.path.exists(model_path): os.makedirs(model_path)
        return img_path, model_path

    def save_checkpoint(self, epoch, best_acc, iteration, is_best=0, gap=1,
                        filename='models/checkpoint.pth.tar', keep_all=False):
        state = {'epoch': epoch + 1,
                 'net': self.args.net,
                 'state_dict': self.model.state_dict(),
                 'best_acc': best_acc,
                 'optimizer': self.optimizer.state_dict(),
                 'iteration': iteration},
        torch.save(state, filename)
        last_epoch_path = os.path.join(os.path.dirname(filename),
                                       'epoch%s.pth.tar' % str(state['epoch'] - gap))
        if not keep_all:
            try:
                os.remove(last_epoch_path)
            except:
                pass
        if is_best:
            past_best = glob.glob(os.path.join(os.path.dirname(filename), 'model_best_*.pth.tar'))
            for i in past_best:
                try:
                    os.remove(i)
                except:
                    pass
            torch.save(state,
                       os.path.join(os.path.dirname(filename), 'model_best_epoch%s.pth.tar' % str(state['epoch'])))

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None

        if self.use_cuda:
            sample = utils.move_to_cuda(sample)

        return sample


def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size()  # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)
