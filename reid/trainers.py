from __future__ import print_function, absolute_import
import time

import torch
from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss
from .utils.meters import AverageMeter

use_gpu = torch.cuda.is_available()

class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        # self.criterion_triple = TripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            t_inputs, _ = self._parse_data(target_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            # target samples: only forward
            t_features, _ = self.model(t_inputs)

            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec

#################### 
class SingelmeanTrainer(object):
    def __init__(self, encoder, num_cluster=700, alpha=0.999):
        super(SingelmeanTrainer, self).__init__()
    
        self.encoder = encoder
        self.num_cluster = num_cluster
        self.alpha = alpha

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda() # margin=0.0 is log-softmax; margin=None uses teacher's predictions as soft labels             
                                                                # attention: our LF2 doesn't use soft labels.
    def train(self, epoch, data_loader_target, loader_upper, loader_low,
            optimizer, print_freq=1, train_iters=200):

        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_tri_2 = [AverageMeter(),AverageMeter()]
        losses_tri_3 = [AverageMeter(),AverageMeter()]
        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            target_global= data_loader_target.next()    # global
            target_upper= loader_upper.next()           # upper
            target_low= loader_low.next()               # low 

            data_time.update(time.time() - end)

            # process inputs
            inputs_1, targets_global = self._parse_data(target_global)
            inputs_2, targets_upper = self._parse_data(target_upper)
            inputs_3, targets_low = self._parse_data(target_low)
            
            # forward
            feat_list, prob, prob_ema, fuse_outputs  = self.encoder(inputs_1)
            feat_list_2, prob_2, prob_2_ema, fuse_outputs_2 = self.encoder(inputs_2)
            feat_list_3, prob_3, prob_3_ema, fuse_outputs_3 = self.encoder(inputs_3)

            loss_ce_1 = 0
            loss_ce_1 += self.criterion_ce(prob, targets_global)

            # global 
            loss_tri_1 = 0
            loss_tri_1 += self.criterion_tri(feat_list[0],feat_list[0], targets_global)
            # upper
            loss_tri_2 = 0
            loss_tri_2 += self.criterion_tri(feat_list_2[1],feat_list_2[1], targets_upper) 
            # low
            loss_tri_3 = 0
            loss_tri_3 += self.criterion_tri(feat_list_3[2],feat_list_3[2], targets_low)
            # total loss
            loss = loss_ce_1 + loss_tri_1 + 0.5*loss_tri_2 + 0.5*loss_tri_3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self._update_ema_variables(self.encoder.model, self.encoder.model_ema, self.alpha, epoch*len(data_loader_target)+i)

            prec_1, = accuracy(prob.data, targets_global.data)

            losses_ce[0].update(loss_ce_1.item())
            losses_tri[0].update(loss_tri_1.item())
            losses_tri_2[0].update(loss_tri_2.item())
            losses_tri_3[0].update(loss_tri_3.item())
            precisions[0].update(prec_1[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\n'
                      'Loss_ce {:.3f}\t'
                      'Loss_tri {:.3f}\t'
                      'Loss_tri_2 {:.3f}\t'
                      'Loss_tri_3 {:.3f}\t'
                      'Prec {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              losses_ce[0].avg,
                              losses_tri[0].avg, 
                              losses_tri_2[0].avg, losses_tri_3[0].avg, 
                              precisions[0].avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)  # ema_param*alpha + (1-alpha)*param

    def _parse_data(self, inputs):
        imgs_1, _, pids, _ = inputs 
        if use_gpu:
            inputs_1 = imgs_1.cuda()
            targets = pids.cuda() 
        return inputs_1, targets
