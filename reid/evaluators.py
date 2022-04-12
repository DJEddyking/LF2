from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import torch

from reid.feature_extraction.cnn import extract_cnn_feature_source
from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking

def fliplr(img):
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_features(model, data_loader, print_freq=50, metric=None, source=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    
    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            if source:
                outputs_1 = extract_cnn_feature_source(model, imgs)
                outputs_2 = extract_cnn_feature_source(model, fliplr(imgs))    
            else:
                outputs_1 = extract_cnn_feature(model, imgs)
                outputs_2 = extract_cnn_feature(model, fliplr(imgs))   
            outputs = outputs_1 + outputs_2
            fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
            outputs = outputs.div(fnorm.expand_as(outputs))
            
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))
    return features, labels



def extract_cnn_feature_new(model, inputs):
    model.eval()
    inputs = inputs.cuda() 
    _, outputs_ = model(inputs)
    outputs = [x.data.cpu() for x in outputs_]  
    return outputs

def extract_features_new(model, data_loader):
    model.eval()
    features = OrderedDict()
    labels = OrderedDict()
    for j, (imgs, fnames, pids, cams) in enumerate(data_loader): 
        outputs = extract_cnn_feature_new(model, imgs)
        outputs_ = extract_cnn_feature_new(model,  fliplr(imgs))
        for i in range(len(outputs)):
            out = outputs[i] + outputs_[i]
            out_norm = torch.norm(out, p=2, dim=1, keepdim=True) 
            out = out.div(out_norm.expand_as(out))
            outputs[i] = out
        for index, (fname, pid, cam) in enumerate(zip(fnames, pids, cams)):
            features[fname] = [x[index] for x in outputs]             
            labels[fname] = pid

    return features, labels 



def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))  
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())  
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)  
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0) 
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t()) 
    return dist_m, x.numpy(), y.numpy()

def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)
                }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k,
                      cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'][0], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False, rerank=False, pre_features=None, source =False):
        if (pre_features is None):
            features, _ = extract_features(self.model, data_loader, source=source)
        else:
            features = pre_features
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery, metric=metric)
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq = pairwise_distance(features, query, query, metric=metric)
        distmat_gg = pairwise_distance(features, gallery, gallery, metric=metric)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)