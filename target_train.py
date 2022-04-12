from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import copy

import sys
sys.path.insert(0, os.getcwd())
sys.path.append('..')

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize

import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader


from reid import datasets
from reid import models
from reid.trainers import SingelmeanTrainer
from reid.evaluators import Evaluator, extract_features_new
from reid.utils.data import IterLoader
from reid.utils.data import transforms as T
from reid.utils.data.sampler import RandomIdentitySampler, RandomMultipleGallerySampler
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.models.resnet import Encoder

start_epoch = best_mAP = 0

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(dataset, height, width, batch_size, workers,
                    num_instances, iters):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])  ## only add in target-domain fine-tuning
         ])

    train_set = dataset.train
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)  
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                        transform=train_transformer, mutual=1),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args, extract_feat_):
    arch = args.arch

    model_student = models.create(arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters, \
                                  num_split=args.split_parts, extract_feat=extract_feat_).cuda()
    model_teacher = models.create(arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters, \
                                  num_split=args.split_parts, extract_feat=extract_feat_).cuda()
    model_student = nn.DataParallel(model_student)  
    model_teacher = nn.DataParallel(model_teacher)

    ## load source-domain pre-training parameters 
    initial_weights = load_checkpoint(osp.join(args.initial_weights,'model_best.pth.tar'))  
    copy_state_dict(initial_weights['state_dict'], model_student)
    copy_state_dict(initial_weights['state_dict'], model_teacher)
    model_teacher.module.classifier.weight.data.copy_(model_student.module.classifier.weight.data)   

    for param in model_teacher.parameters():
        param.detach_()

    return model_student, model_teacher

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    iters = args.iters if (args.iters>0) else None

    # Create target domain test loader
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)

    # Create teacher-student encoder
    model_student, model_teacher = create_model(args, False)
    encoder = Encoder(model_student, model_teacher)
    
    if args.resume:
        encoder_weights = load_checkpoint(osp.join(args.resume, 'model_best.pth.tar'))
        copy_state_dict(encoder_weights['state_dict'], encoder)

    # Evaluator
    evaluator_ = Evaluator(encoder)

    clusters = [args.num_clusters]*args.epochs
    for nc in range(len(clusters)):
        # Create target domain train loader
        cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers, testset=dataset_target.train)

        # extract features
        features, _ = extract_features_new(encoder, cluster_loader)  
        len_f = len(features[dataset_target.train[0][0]])     
        features = [torch.cat([features[f][i].unsqueeze(0) for f, _, _ in dataset_target.train], 0) for i in range(len_f)]
        
        cf_global =  features[0]    # global features
        cf_upper = features[1]      # upper features
        cf_low = features[2]        # low features

        # Clustering
        print('\n Clustering into {} classes \n'.format(clusters[nc]))
        km_global = MiniBatchKMeans(n_clusters=clusters[nc], max_iter=100, batch_size=300, init_size=1500).fit(cf_global)  
        km_upper = MiniBatchKMeans(n_clusters=clusters[nc], max_iter=100, batch_size=300, init_size=900).fit(cf_upper)
        km_low = MiniBatchKMeans(n_clusters=clusters[nc], max_iter=100, batch_size=300, init_size=900).fit(cf_low)
        # update classifier
        encoder.model.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km_global.cluster_centers_, axis=1)).float().cuda()) 
        encoder.model_ema.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km_global.cluster_centers_, axis=1)).float().cuda()) 
        # get clutering labels
        target_label_global = km_global.labels_
        target_label_upper = km_upper.labels_
        target_label_low = km_low.labels_

        # Create new datasets
        dataset_target_upper = copy.deepcopy(dataset_target)
        dataset_target_low = copy.deepcopy(dataset_target)
        for i in range(len(dataset_target.train)):  
            dataset_target.train[i] = list(dataset_target.train[i])
            dataset_target.train[i][1] = int(target_label_global[i])
            dataset_target.train[i] = tuple(dataset_target.train[i])

        for i in range(len(dataset_target_upper.train)):  
            dataset_target_upper.train[i] = list(dataset_target_upper.train[i])
            dataset_target_upper.train[i][1] =  int(target_label_upper[i])
            dataset_target_upper.train[i] = tuple(dataset_target_upper.train[i])

        for i in range(len(dataset_target_low.train)):  
            dataset_target_low.train[i] = list(dataset_target_low.train[i])
            dataset_target_low.train[i][1] = int(target_label_low[i])
            dataset_target_low.train[i] = tuple(dataset_target_low.train[i])

        # Create new dataloaders
        train_loader_target = get_train_loader(dataset_target, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters)

        train_loader_target_upper = get_train_loader(dataset_target_upper, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters)

        train_loader_target_low = get_train_loader(dataset_target_low, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters)

        # Optimizer
        params = []
        for key, value in encoder.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
        optimizer = torch.optim.Adam(params)

        # Trainer
        trainer = SingelmeanTrainer(encoder, num_cluster=clusters[nc], alpha=args.alpha)
        train_loader_target.new_epoch() 
        epoch = nc
        trainer.train(epoch, train_loader_target, train_loader_target_upper, train_loader_target_low, optimizer,
                      print_freq=args.print_freq, train_iters=args.iters)

        # Save best model
        def save_model(model_ema, is_best, best_mAP, mid):
            save_checkpoint({
                'state_dict': model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'model'+str(mid)+'_checkpoint.pth.tar'))

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            mAP = evaluator_.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False, source=False)
            is_best = mAP > best_mAP
            best_mAP = max([mAP] + [best_mAP])  
            save_model(encoder, (is_best and (mAP==best_mAP)), best_mAP, '_last')

            print('\n * Finished epoch {:3d}  model no.1 mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))  

    # Test on target domain
    print ('\n Test on the best model.\n')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    encoder.load_state_dict(checkpoint['state_dict'])
    evaluator_.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True, source=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LF2 Training")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str, default='market') # target domain
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--num-clusters', type=int, default=700)    # clusters number
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50')
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--split-parts', type=int, default=2)       # splitted parts
    parser.add_argument('--initial-weights', type=str, metavar='PATH',\
                        default='logs_duke_pretrained_multi_gpu')
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)        # temporal ensemble momentum
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=80) 
    parser.add_argument('--iters', type=int, default=400)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--eval-step', type=int, default=1) 
    parser.add_argument('--resume', type=str, metavar='PATH', default='')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(os.getcwd(),'/home/dj/reid/data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(os.getcwd(), 'logs_d2m')) 
    main()
