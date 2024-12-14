import argparse
import os, sys
current_dir = os.path.dirname((os.path.abspath(__file__)))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import os.path as osp
import torchvision
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from typing import Optional, List
from PIL import Image, ImageFile

# import torchvision.transforms.functional as TF

# from analysis import collect_feature, tsne, a_distance

class RandomRGBShift(transforms.ColorJitter):
    def __init__(self, shift_intensity=0.1):
        super(RandomRGBShift, self).__init__()
        self.shift_intensity = shift_intensity

    def __call__(self, img):
        # 为每个通道生成一个随机偏移量
        shifts = torch.randn(3) * self.shift_intensity

        # 分别对R, G, B通道应用偏移量
        r, g, b = img.split()
        r = TF.adjust_brightness(r, 1 + shifts[0])
        g = TF.adjust_brightness(g, 1 + shifts[1])
        b = TF.adjust_brightness(b, 1 + shifts[2])

        # 重组图像
        img = transforms.functional.to_pil_image(torch.stack((r, g, b)))
        return img



class AverageMeter(object):
    r"""Computes and stores the average and current value.

    Examples::

        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """
    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # RandomRGBShift(shift_intensity=0.05),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    # txt_src = open(args.s_dset_path).readlines()
    # txt_test = open(args.test_dset_path).readlines()
    #
    # dsize = len(txt_src)
    # tr_size = int(0.9*dsize)
    # print(dsize, tr_size, dsize - tr_size)
    # tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    # path = '/nfs/ofs-902-1/object-detection/zhujiankun/EDA/code/SHOT/list_file/Emo_Rec/image_list'
    # tr_txt = os.path.join(path,f'{args.dset}_train.txt')
    # te_txt = os.path.join(path,f'{args.dset}_test.txt')
    dsets["source_tr"] = ImageList(open(args.tr_txt).readlines(), transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(open(args.te_txt).readlines(), transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs*3, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(open(args.test_dset_path).readlines(), transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=True,
        num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.metrics import classification_report
    amusement = AverageMeter('amusement', ':6.2f')
    awe = AverageMeter('awe', ':6.2f')
    contentment = AverageMeter('contentment', ':6.2f')
    disgust = AverageMeter('disgust', ':6.2f')
    excitement = AverageMeter('excitement', ':6.2f')
    fear = AverageMeter('fear', ':6.2f')
    sadness = AverageMeter('sadness', ':6.2f')
    anger = AverageMeter('anger', ':6.2f')
    start_test = True
    with torch.no_grad():
        taret_names = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            target1 = labels.cpu()
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            _, pred = outputs.topk(1, 1, True, True)
            pred1 = pred.cpu().numpy()
            report_dict = classification_report(target1, pred1, target_names=taret_names, labels=range(8),
                                                output_dict=True)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
            amusement.update(report_dict['amusement']['precision'], inputs.size(0))
            anger.update(report_dict['anger']['precision'], inputs.size(0))
            awe.update(report_dict['awe']['precision'], inputs.size(0))
            contentment.update(report_dict['contentment']['precision'], inputs.size(0))
            disgust.update(report_dict['disgust']['precision'], inputs.size(0))
            excitement.update(report_dict['excitement']['precision'], inputs.size(0))
            fear.update(report_dict['fear']['precision'], inputs.size(0))
            sadness.update(report_dict['sadness']['precision'], inputs.size(0))

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
    all_label1 = all_label.cpu().numpy().tolist()
    predict1 = predict.cpu().numpy().tolist()
    report_dict = classification_report(all_label1, predict1, target_names=taret_names, labels=range(8),
                                        output_dict=True)
    print('-----------------cal_acc--------------------')
    print('amusement', report_dict['amusement'])
    print('anger', report_dict['anger'])
    print('awe', report_dict['awe'])
    print('contentment', report_dict['contentment'])
    print('disgust', report_dict['disgust'])
    print('excitement', report_dict['excitement'])
    print('fear', report_dict['fear'])
    print('sadness', report_dict['sadness'])
    print('macro avg', report_dict['macro avg'])
    print('weighted avg', report_dict['weighted avg'])
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        print('matrix1',matrix)
        matrix = matrix[np.unique(all_label).astype(int),:]
        print('matrix2',matrix)
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        print('aacc, acc',aacc, acc)
        return accuracy * 100, mean_ent
    else:
        return accuracy*100, mean_ent


def map_classes_to_binary(classes):
    return [0 if x in [0, 2, 3, 5] else 1 for x in classes]


def cal_acc_two(loader, netF, netB, netC, flag=False):
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.metrics import classification_report
    start_test = True
    with torch.no_grad():
        taret_names = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            _, pred = outputs.topk(1, 1, True, True)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    binary_target = map_classes_to_binary(all_label)
    print('predict', predict)
    binary_pred = map_classes_to_binary(predict)
    print('binary_pred', binary_pred)
    binary_acc = sum(1 for a,b in zip(binary_target,binary_pred)if a == b) / len(binary_target)
    print(f'binary acc:{binary_acc}')
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
    all_label1 = all_label.cpu().numpy().tolist()
    predict1 = predict.cpu().numpy().tolist()
    report_dict = classification_report(all_label1, predict1, target_names=taret_names, labels=range(8),
                                        output_dict=True)
    print('-----------------cal_acc--------------------')
    print('amusement', report_dict['amusement'])
    print('anger', report_dict['anger'])
    print('awe', report_dict['awe'])
    print('contentment', report_dict['contentment'])
    print('disgust', report_dict['disgust'])
    print('excitement', report_dict['excitement'])
    print('fear', report_dict['fear'])
    print('sadness', report_dict['sadness'])
    print('macro avg', report_dict['macro avg'])
    print('weighted avg', report_dict['weighted avg'])

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        print('matrix1', matrix)
        matrix = matrix[np.unique(all_label).astype(int), :]
        print('matrix2', matrix)
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        print('aacc, acc', aacc, acc)
        return accuracy * 100, mean_ent
    else:
        return accuracy * 100, mean_ent



def kd_loss_function(output, target_output,args):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / args.temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd



def train_source(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    elif args.net == 'vit':
        print('!!!!!!!!!!!!VVVVIIIIIITTTTTT')
        netF = network.ViT().cuda()
        # print('000')
    elif args.net == 'senet':
        print('senet')
        netF = network.SENET().cuda()
    elif args.net == 'byot':
        print('byot')
        netF = network.ResBase_BYOT(res_name=args.net).cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    # todo!!!!!!!
    # optimizer = optim.AdamW(param_group)
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    criterion = nn.CrossEntropyLoss().cuda()

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    # print('11')
    netF.train()
    # print('22')
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = next(iter_source)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        # print('inputs_source',inputs_source.shape)
        output_source = netF(inputs_source)
        outputs_source = netC(netB(output_source))
        # print('outputs_source',outputs_source.shape)
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()


        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()
                torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
                torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
                torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))
            torch.save(netF, osp.join(args.output_dir_src, f"source_F_{iter_num}.pt"))
            torch.save(netB, osp.join(args.output_dir_src, f"source_B_{iter_num}.pt"))
            torch.save(netC, osp.join(args.output_dir_src, f"source_C_{iter_num}.pt"))

            netF.train()
            netB.train()
            netC.train()
                
    # torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    # torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
    # torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF, netB, netC

def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  
    elif args.net == 'vit':
        print('!!!!!!!!!!!!VVVVIIIIIITTTTTT')
        netF = network.ViT().cuda()
    elif args.net == 'senet':
        print('senet')
        netF = network.SENET().cuda()
    elif args.net == 'byot':
        print('byot')
        netF = network.ResBase_BYOT(res_name=args.net).cuda()


    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    
    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    feature_extractor = nn.Sequential(
        netF).to('cuda')

    # source_feature, target_feature = collect_feature(
    #     dset_loaders['source_te'], feature_extractor, 'cuda')
    # target_feature = collect_feature(
    #     dset_loaders['test'], feature_extractor, 'cuda', max_num_features=10)
    # plot t-SNE
    # tSNE_filename = osp.join('DBSCAN', 'TSNE_dbscan20.pdf')
    # tsne.visualize(source_feature, target_feature, tSNE_filename)
    # print("Saving t-SNE to", tSNE_filename)
    # calculate A-distance, which is a measure for distribution discrepancy
    # A_distance = a_distance.calculate(
    #     source_feature, target_feature, 'cuda')
    # print("A-distance =", A_distance)

    if args.dset=='VISDA-C':
        acc, acc_list = cal_acc_two(dset_loaders['test'], netF, netB, netC, True)
        log_str = '\nTask: {}, Accuracy = {:.2f}%'.format(args.name, acc) + '\n' + acc_list
    else:
        acc, _ = cal_acc_cdd(dset_loaders['test'], netF, netB, netC, False)
        # acc, _ = cal_acc_two(dset_loaders['test'], netF, netB, netC, False)
        log_str = '\nTask: {}, Accuracy = {:.2f}%'.format(args.name, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def cal_acc_cdd(loader, netF, netB, netC, flag=False):
    import torch.nn.functional as F

    start_test = True
    warnings.filterwarnings('ignore')

    # Storing results for classification report
    all_output = []
    all_label = []
    features_by_class = {i: [] for i in range(8)}  # Assuming 8 classes
    centers = torch.zeros(8, 256).cuda()  # Assuming 512-dimensional feature space, adjust if needed

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()

            # Get the feature embeddings before the final classification layer (netB(netF(inputs)))
            features = netB(netF(inputs))  # Shape: [batch_size, feature_dim] (e.g., [192, 512])

            # Classifier output (optional for classification report)
            outputs = netC(features)  # Classifier output, shape: [batch_size, num_classes]

            # Collect all features and labels for report and CCD computation
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

            # Collect features for each class (needed for CCD and PDD)
            for i in range(len(labels)):
                features_by_class[labels[i].item()].append(features[i].cpu().numpy())

    # Convert all_output to numpy for classification report
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    # Compute class centers (mean of features in each class)
    for class_id in range(8):  # Assuming 8 classes
        class_features = torch.tensor(features_by_class[class_id]).cuda()  # [num_samples, feature_dim]

        # Ensure the class features have the correct shape [num_samples, feature_dim]
        centers[class_id] = class_features.mean(dim=0)  # Mean along dimension 0 (samples), result: [feature_dim]

    # Calculate CCD
    intra_class_compactness = torch.zeros(8).cuda()
    inter_class_distance = torch.zeros(8, 8).cuda()
    ccd_per_class = torch.zeros(8).cuda()  # To store the CCD for each class

    # Intra-class compactness: sum of squared distances to center for each class
    for class_id in range(8):
        class_features = torch.tensor(features_by_class[class_id]).cuda()
        center = centers[class_id]
        intra_class_compactness[class_id] = torch.mean(torch.sum((class_features - center) ** 2, dim=1))

    # Inter-class distance: squared Euclidean distance between each pair of class centers
    for i in range(8):
        for j in range(i + 1, 8):
            inter_class_distance[i, j] = torch.sum((centers[i] - centers[j]) ** 2)
            inter_class_distance[j, i] = inter_class_distance[i, j]

    # CCD calculation: for each class, compute the ratio of intra-class compactness to inter-class distance
    C = 8  # Total number of classes
    epsilon = 1e-8  # Small value to prevent division by zero
    for i in range(C):
        ccd_sum = 0
        for j in range(C):
            if i != j:
                inter_dist = inter_class_distance[i, j]
                if inter_dist == 0:
                    inter_dist = epsilon  # Prevent division by zero
                ratio = intra_class_compactness[i] / inter_dist
                ccd_sum += ratio
        ccd_per_class[i] = ccd_sum / (C - 1)

    # Calculate average CCD
    avg_ccd = torch.mean(ccd_per_class).item()

    # PDD Calculation
    pdd_per_class = torch.zeros(8).cuda()  # To store the PDD for each class
    epsilon = 1e-8  # A small constant to prevent division by zero

    for class_id in range(8):
        class_features = torch.tensor(features_by_class[class_id]).cuda()  # [num_samples, feature_dim]
        center_i = centers[class_id]  # The center of class_i
        pdd_sum = 0
        # Iterate over all features in class_i
        for x in class_features:
            # Cosine similarity between x and the class center
            cos_i = F.cosine_similarity(x.unsqueeze(0), center_i.unsqueeze(0), dim=1)  # Cos(x, mu_i)
            cos_i = (cos_i + 1) / 2  # Map to [0, 1]

            # Sum of cosine similarities between x and all other class centers
            cos_sum = 0
            for other_class_id in range(8):
                if other_class_id != class_id:
                    center_j = centers[other_class_id]  # The center of class_j
                    cos_j = F.cosine_similarity(x.unsqueeze(0), center_j.unsqueeze(0), dim=1)  # Cos(x, mu_j)
                    cos_j = (cos_j + 1) / 2  # Map to [0, 1]
                    cos_sum += cos_j

            cos_sum = cos_sum + epsilon  # Prevent division by zero

            pdd_sample = cos_i / cos_sum  # Compute the ratio for each sample
            pdd_sum += pdd_sample.item()  # Add to the sum

        # Average PDD for this class
        pdd_per_class[class_id] = 1 - (pdd_sum / class_features.size(0))

        # Ensure PDD is within [0, 1]
        pdd_per_class[class_id] = torch.clamp(pdd_per_class[class_id], min=0.0, max=1.0)

    # Calculate average PDD
    avg_pdd = torch.mean(pdd_per_class).item()

    # Print classification report
    target_names = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
    all_label = all_label.cpu().numpy().tolist()
    predict = predict.cpu().numpy().tolist()
    report_dict = classification_report(all_label, predict, target_names=target_names, labels=range(8),
                                        output_dict=True)

    print('-----------------cal_acc--------------------')
    for label in target_names:
        print(f'{label}: {report_dict[label]}')
    print('macro avg', report_dict['macro avg'])
    print('weighted avg', report_dict['weighted avg'])

    # 输出CCD和PDD值
    print('-----------------CCD and PDD--------------------')
    for i, label in enumerate(target_names):
        print(f'Class {label}: CCD = {ccd_per_class[i].item():.4f}, PDD = {pdd_per_class[i].item():.4f}')

    print(f'Average CCD: {avg_ccd:.4f}')
    print(f'Average PDD: {avg_pdd:.4f}')

    # Optionally, return confusion matrix or other metricsdd
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        mean_ent = torch.mean(F.cross_entropy(all_output, all_label.long(), reduction='none')).cpu().data.item()
        ccd_values = ccd_per_class.cpu().numpy().tolist()  # Convert CCD tensor to list
        pdd_values = pdd_per_class.cpu().numpy().tolist()  # Convert PDD tensor to list
        return accuracy * 100, mean_ent



if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description='SHOT++')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=1, help="source")
    parser.add_argument('--t', type=int, default=0, help="target")
    parser.add_argument('--max_epoch', type=int, default=50, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=0, help="number of workers")
    parser.add_argument('--dset', type=str, default='Emo_Rec', choices=['VISDA-C', 'office', 'office-home','Emo_Rec','Art_FI','E6_FI'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--net', type=str, default='resnet101', help="vgg16, resnet18, resnet34, resnet50, resnet101, vit, senet")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    # todo!!!!
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda'])
    parser.add_argument('--temperature', default=3, type=int,
                        help='temperature to smooth the logits')
    parser.add_argument('--alpha', default=0.1, type=float,
                        help='weight of kd loss')
    parser.add_argument('--beta', default=1e-6, type=float,
                        help='weight of feature loss')
    args = parser.parse_args()
    print('s',args.s)
    print('t',args.t)
    print('--lr',args.lr)

    if args.dset == 'office-home':
        names = ['RealWorld','Clipart']
        args.class_num = 8
    elif args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    elif args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
        args.lr = 1e-3
    elif args.dset == 'Art_FI':
        names = ['FI', 'Art']
        args.class_num = 8
    elif args.dset == 'E6_FI':
        names = ['FI', 'E6']
        args.class_num = 2
    elif args.dset == 'Emo_Rec':
        names = ['FI', 'EmoSet']
        args.class_num = 8

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = '/home/cx_xchen/zhuzhu/apps/Emo-SFDA/BBA/list_file/EmoSet_FI'
    args.test_dset_path = folder + '/' + names[args.t] + '_test.txt'
    args.tr_txt = os.path.join(folder, f'{names[args.s]}_train.txt')
    args.te_txt = os.path.join(folder, f'{names[args.s]}_test.txt')
    # args.tr_txt = os.path.join(folder, f'train_split_{jj}.txt')
    # args.te_txt = os.path.join(folder, f'test_split_{jj}.txt')

    args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper())

    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    # train_source(args)


    args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        args.name = names[args.s][0].upper() + names[args.t][0].upper()
        test_target(args)


    args.out_file.close()
