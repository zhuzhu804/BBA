import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth,KnowledgeDistillationLoss
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from Modules.masking import Masking
import torch.nn.functional as F


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
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
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

    dsets["target"] = ImageList_idx(open(args.tr_txt).readlines(), transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(open(args.te_txt).readlines(), transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=False,
                                      num_workers=args.worker, drop_last=False)
    dsets["target_te"] = ImageList_idx(open(args.tr_txt).readlines(), transform=image_test())
    dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=train_bs, shuffle=False,
                                           num_workers=args.worker, drop_last=False)
    # dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    # dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    # dsets["source_te"] = ImageList(te_txt, transform=image_test())
    # dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    # dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    # dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    # dsets["target_te"] = ImageList(txt_tar, transform=image_test())
    # dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False)
    # dsets["test"] = ImageList(txt_test, transform=image_test())
    # dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.metrics import classification_report
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if netB is None:
                outputs = netC(netF(inputs))
            else:
                outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item() / np.log(all_label.size()[0])
    taret_names = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
    all_label = all_label.cpu().numpy().tolist()
    predict = predict.cpu().numpy().tolist()
    report_dict = classification_report(all_label, predict, target_names=taret_names, labels=range(8),
                                        output_dict=True)

    # print(report_dict)
    # raise Exception
    print('-----------------cal_acc--------------------')
    print('amusement', report_dict['amusement'])
    print('anger', report_dict['anger'])
    print('awe', report_dict['awe'])
    print('contentment', report_dict['contentment'])
    print('disgust', report_dict['disgust'])
    print('excitement', report_dict['excitement'])
    print('fear', report_dict['fear'])
    print('sadness', report_dict['sadness'])
    print('macro avg',report_dict['macro avg'])
    print('weighted avg',report_dict['weighted avg'])

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int),:]
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc, mean_ent
    else:
        return accuracy*100, mean_ent

def train_source_simp(args):
    dset_loaders = data_load(args)
    if args.net_src[0:3] == 'res':
        netF = network.ResBase(res_name=args.net_src).cuda()
    # netC = network.feat_classifier_simpl(class_num=args.class_num, feat_dim=netF.in_features).cuda()
    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer,class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr_src
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 1}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netC.train()

    netB.train()

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
        outputs_source = netC(netB(netF(inputs_source)))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1)(outputs_source, labels_source)            
        
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netC.eval()
            netB.eval()
            acc_s_te, _ = cal_acc_two(dset_loaders['source_te'], netF, netB, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netC = netC.state_dict()
                best_netB = netB.state_dict()

            netF.train()
            netC.train()
            netB.train()
                
    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))
    torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))

    return netF, netB,netC

def test_target_simp(args):
    dset_loaders = data_load(args)
    if args.net_src[0:3] == 'res':
        netF = network.ResBase(res_name=args.net_src).cuda()
    # netC = network.feat_classifier_simpl(class_num = args.class_num, feat_dim=netF.in_features).cuda()
    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    
    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, _ = cal_acc_two(dset_loaders['test'], netF, netB, netC, False)
    log_str = '\nTask: {}, Accuracy = {:.2f}%'.format(args.name, acc)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')


def polarity_loss(outputs, labels):
    import torch.nn.functional as F
    # 定义积极和消极类别
    positive_classes = [0, 2, 3, 5]

    # 创建积极的掩码
    pos_mask = torch.zeros_like(outputs)
    for c in positive_classes:
        pos_mask[:, c] = 1

    # 计算积极的预测概率
    pos_probs = F.softmax(outputs, dim=1) * pos_mask
    pos_probs = pos_probs.sum(dim=1)

    # 构建情感极性的二分类标签
    polarity_labels = torch.zeros_like(labels, dtype=torch.float)
    for c in positive_classes:
        polarity_labels[labels == c] = 1

    # 计算情感极性的交叉熵损失
    return F.binary_cross_entropy(pos_probs, polarity_labels)


class ModifiedKLDivergenceLoss(nn.Module):
    def __init__(self, positive_labels, negative_labels, large_penalty):
        super(ModifiedKLDivergenceLoss, self).__init__()
        self.positive_labels = positive_labels.tolist()  # Convert to list for compatibility
        self.negative_labels = negative_labels.tolist()  # Convert to list for compatibility
        self.large_penalty = large_penalty

    def forward(self, predicted_probabilities, true_probabilities):
        # Calculate the KL divergence loss
        kl_div_loss = F.kl_div(predicted_probabilities.log(), true_probabilities, reduction='none')

        # Determine the true and predicted classes from probabilities
        _, predicted_classes = torch.max(predicted_probabilities, 1)
        _, true_classes = torch.max(true_probabilities, 1)

        # Check if the polarity of the prediction is incorrect
        true_polarity = torch.tensor([1 if label in self.positive_labels else 0 for label in true_classes])
        predicted_polarity = torch.tensor([1 if label in self.positive_labels else 0 for label in predicted_classes])
        polarity_true = true_polarity == predicted_polarity

        # Apply a larger penalty when the polarity is incorrect
        penalty = torch.ones_like(kl_div_loss.sum(dim=1))
        penalty[polarity_true] = self.large_penalty
        kl_div_loss = kl_div_loss.sum(dim=1) * penalty

        return kl_div_loss.mean()


def copy_target_simp(args):
    dset_loaders = data_load(args)
    if args.net_src[0:3] == 'res':
        netF = network.ResBase(res_name=args.net_src).cuda()
    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                       bottleneck_dim=args.bottleneck).cuda()
    # netC = network.feat_classifier_simpl(class_num=args.class_num, feat_dim=netF.in_features).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    source_model = nn.Sequential(netF, netB, netC).cuda()
    source_model.eval()

    if args.net[0:3] == 'res':
        netF_s = network.ResBase(res_name=args.net).cuda()
    netB_s = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC_s = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    # args.modelpath = args.output_dir_src + '/source_F.pt'
    # netF_s.load_state_dict(torch.load(args.modelpath))
    # modelpath = args.output_dir_src + '/source_B.pt'
    # netB_s.load_state_dict(torch.load(modelpath))
    # args.modelpath = args.output_dir_src + '/source_C.pt'
    # netC_s.load_state_dict(torch.load(args.modelpath))


    param_group = []
    learning_rate = args.lr
    for k, v in netF_s.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB_s.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC_s.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    ent_best = 1.0
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // 10
    iter_num = 0
    initial_m = 0.9
    finel_m = 0.001  # momentum parameter
    total_ep = max_iter // 2
    m_decay = (initial_m - finel_m) / total_ep

    model = nn.Sequential(netF_s, netB_s, netC_s).cuda()
    model.eval()

    start_test = True
    with torch.no_grad():
        iter_test = iter(dset_loaders["target_te"])
        for i in range(len(dset_loaders["target_te"])):
            data = next(iter_test)
            inputs, labels = data[0], data[1]
            inputs = inputs.cuda()
            outputs = source_model(inputs)
            outputs = nn.Softmax(dim=1)(outputs)
            _, src_idx = torch.sort(outputs, 1, descending=True)
            # if args.topk > 0:
            #     topk = np.min([args.topk, args.class_num])
            #     for i in range(outputs.size()[0]):
            #         outputs[i, src_idx[i, topk:]] = (1.0 - outputs[i, src_idx[i, :topk]].sum())/ (outputs.size()[1] - topk)

            if start_test:
                all_output = outputs.float()
                all_label = labels
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels), 0)
        mem_P = all_output.detach()

    masking = Masking(
        block_size=args.mask_block_size,
        ratio=args.mask_ratio,
        color_jitter_s=args.mask_color_jitter_s,
        color_jitter_p=args.mask_color_jitter_p,
        blur=args.mask_blur,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))
    acc_init = 0.0
    model.train()
    mem_label = obtain_label_without_clu(dset_loaders['test'], netF, netB, netC, args).cuda()
    while iter_num < max_iter:
        # if iter_num % interval_iter == 0:
        #     # mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
        #     mem_label = obtain_label_without_clu(dset_loaders['test'], netF, netB, netC, args).cuda()
        #     # mem_label = torch.from_numpy(mem_label).cuda()
        if iter_num > 0 and iter_num % interval_iter == 0:
            epoch_ema = iter_num // interval_iter
            print('epoch_ema', epoch_ema)
            lambda_i = args.ema * torch.exp(-torch.tensor(epoch_ema / 6))
            print('lambda_i', lambda_i)
            model.eval()
            start_test = True
            with torch.no_grad():
                iter_test = iter(dset_loaders["target_te"])
                for i in range(len(dset_loaders["target_te"])):
                    data = next(iter_test)
                    inputs = data[0]
                    inputs = inputs.cuda()
                    outputs = model(inputs)
                    outputs = nn.Softmax(dim=1)(outputs)
                    if start_test:
                        all_output = outputs.float()
                        start_test = False
                    else:
                        all_output = torch.cat((all_output, outputs.float()), 0)
                mem_P = mem_P * lambda_i + all_output.detach() * (1 - lambda_i)
                # mem_P = mem_P * args.ema + all_output.detach() * (1 - args.ema)

            model.train()


        try:
            inputs_target, y, tar_idx = next(iter_target)
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_target, y, tar_idx = next(iter_target)

        if inputs_target.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter, power=1.5)
        inputs_target = inputs_target.cuda()
        with torch.no_grad():
            outputs_target_by_source = mem_P[tar_idx, :]
            _, src_idx = torch.sort(outputs_target_by_source, 1, descending=True)
        outputs_target = model(inputs_target)
        outputs_target1 = outputs_target
        outputs_target = torch.nn.Softmax(dim=1)(outputs_target)
        classifier_loss = nn.KLDivLoss(reduction='batchmean')(outputs_target.log(), outputs_target_by_source)
        cross_entropy_loss_fn = nn.CrossEntropyLoss()
        # 计算交叉熵损失，使用伪标签src_idx[:, 0]作为真实标签
        # try:
        #     cross_entropy_loss = cross_entropy_loss_fn(outputs_target1, src_idx[:, 0])
        #     classifier_loss += cross_entropy_loss
        #     binary_loss = polarity_loss(outputs_target1, src_idx[:, 0])
        #     classifier_loss += binary_loss
        # except:
        #     print(outputs_target1)
        #     raise Exception

        pred = mem_label[tar_idx]
        classifier_loss += nn.CrossEntropyLoss()(outputs_target1, pred)
        # classifier_loss += adaptive_entropy_loss(outputs_target1, pred)
        # binary_loss = polarity_loss(outputs_target, pred)
        # classifier_loss += binary_loss

        optimizer.zero_grad()

        entropy_loss = torch.mean(loss.Entropy(outputs_target))
        msoftmax = outputs_target.mean(dim=0)
        gentropy_loss = torch.sum(- msoftmax * torch.log(msoftmax + 1e-5))
        entropy_loss -= gentropy_loss
        classifier_loss += entropy_loss

        classifier_loss.backward()

        # if args.mix > 0:
        #     alpha = 0.3
        #     lam = np.random.beta(alpha, alpha)
        #     index = torch.randperm(inputs_target.size()[0]).cuda()
        #     mixed_input = lam * inputs_target + (1 - lam) * inputs_target[index, :]
        #     mixed_output = (lam * outputs_target + (1 - lam) * outputs_target[index, :]).detach()
        #
        #     update_batch_stats(model, False)
        #     outputs_target_m = model(mixed_input)
        #     update_batch_stats(model, True)
        #     outputs_target_m = torch.nn.Softmax(dim=1)(outputs_target_m)
        #     classifier_loss = args.mix*nn.KLDivLoss(reduction='batchmean')(outputs_target_m.log(), mixed_output)
        #     classifier_loss.backward()
        optimizer.step()
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()
            acc_s_te, mean_ent = cal_acc_two(dset_loaders['test'], netF_s, netB_s, netC_s, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%, Ent = {:.4f}'.format(args.name, iter_num, max_iter, acc_s_te, mean_ent)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            model.train()
            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF_s.state_dict()
                best_netC = netC_s.state_dict()
                best_netB = netB_s.state_dict()
                torch.save(best_netF, osp.join(args.output_dir, "source_F.pt"))
                torch.save(best_netB, osp.join(args.output_dir, "source_B.pt"))
                torch.save(best_netC, osp.join(args.output_dir, "source_C.pt"))


def adaptive_entropy_loss(outputs, targets):
    # 计算基础的交叉熵损失
    base_loss = F.cross_entropy(outputs, targets, reduction='none')  # 使用 'none' 以获得每个样本的损失

    # 计算置信度（Softmax输出的最大值）
    confidence = torch.max(F.softmax(outputs, dim=1), dim=1)[0]

    # 定义置信度阈值
    confidence_threshold = 0.8

    # 计算权重：对置信度高于阈值的样本减小权重
    weights = torch.where(confidence > confidence_threshold, 0.5 * torch.ones_like(confidence), torch.ones_like(confidence))

    # 计算自适应损失
    adaptive_loss = torch.mean(base_loss * weights)

    return adaptive_loss


def obtain_label_without_clu(loader, netF, netB, netC, args):
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.metrics import classification_report
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)

    _, predict = torch.max(all_output, 1)
    binary_target = map_classes_to_binary(all_label)
    binary_pred = map_classes_to_binary(predict)

    binary_acc = sum(1 for a, b in zip(binary_target, binary_pred) if a == b) / len(binary_target)
    print(f'binary acc:{binary_acc}')

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    print(f'acc:{accuracy}')
    taret_names = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
    all_label = all_label.cpu().numpy().tolist()
    pred_label1 = torch.squeeze(predict).float().tolist()
    report_dict = classification_report(all_label, pred_label1, target_names=taret_names, labels=range(8),
                                        output_dict=True)

    # print(report_dict)
    # raise Exception
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


    return torch.squeeze(predict)



def obtain_label(loader, netF, netB, netC, args):
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.metrics import classification_report
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)

    _, predict = torch.max(all_output, 1)
    binary_target = map_classes_to_binary(all_label)
    binary_pred = map_classes_to_binary(predict)

    binary_acc = sum(1 for a, b in zip(binary_target, binary_pred) if a == b) / len(binary_target)
    print(f'binary acc:{binary_acc}')

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    ### all_fea: extractor feature [bs,N]
    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    ### aff: softmax output [bs,c]

    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    taret_names = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
    all_label = all_label.cpu().numpy().tolist()
    pred_label1 = pred_label.tolist()
    report_dict = classification_report(all_label, pred_label1, target_names=taret_names, labels=range(8),
                                        output_dict=True)

    # print(report_dict)
    # raise Exception
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
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return pred_label.astype('int')


def map_classes_to_binary(classes):
    # print(classes)
    return [0 if x in [0, 2, 3, 5] else 1 for x in classes]


def update_batch_stats(model, flag):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.update_batch_stats = flag

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


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
    binary_pred = map_classes_to_binary(predict)

    binary_acc = sum(1 for a,b in zip(binary_target,binary_pred)if a == b) / len(binary_target)
    print(f'binary acc:{binary_acc}')
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
    all_label1 = all_label.cpu().numpy().tolist()
    predict1 = predict.cpu().numpy().tolist()
    report_dict = classification_report(all_label1, predict1, target_names=taret_names, labels=range(8),
                                        output_dict=True)
    print('-----------------cal_acc-two--------------------')
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DINE')
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=1, help="source")
    parser.add_argument('--t', type=int, default=0, help="target")
    parser.add_argument('--max_epoch', type=int, default=100, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'image-clef', 'office-home', 'office-caltech', 'Emo_Rec', 'Art_FI', 'E6_FI'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101', help="alexnet, vgg16, resnet18, resnet34, resnet50, resnet101")
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--lr_src', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net_src', type=str, default='resnet101', help="alexnet, vgg16, resnet18, resnet34, resnet50, resnet101")
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--ema', type=float, default=1.2)
    parser.add_argument('--mix', type=float, default=1.0)
    parser.add_argument('--mask_block_size', default=32, type=int)
    parser.add_argument('--mask_ratio', default=0.5, type=float)
    parser.add_argument('--mask_color_jitter_s', default=0, type=float)
    parser.add_argument('--mask_color_jitter_p', default=0, type=float)
    parser.add_argument('--mask_blur', default=False, type=bool)
    args = parser.parse_args()
    print('lr', args.lr)
    print('s', args.s)
    print('t', args.t)
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'Emo_Rec':
        names = ['FI', 'EmoSet']
        args.class_num = 8
    if args.dset == 'Art_FI':
        names = ['FI', 'Art']
        args.class_num = 8
    elif args.dset == 'E6_FI':
        names = ['FI', 'E6']
        args.class_num = 2



    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    # for jj in range(8):
    #     print('____' * 20)
    #     print(jj)
    #     print('____' * 20)
    folder = '/home/cx_xchen/zhuzhu/apps/Emo-SFDA/BBA/list_file/Emotion6_FI'
    # args.test_dset_path = folder + '/' + names[args.t] + '_test.txt'
    args.tr_txt = os.path.join(folder, f'{names[args.t]}_test.txt')
    args.te_txt = os.path.join(folder, f'{names[args.t]}_test.txt')

    # args.output_dir_src = osp.join(args.output_src, 'uda', args.dset, names[args.s][0].upper())
    args.output_dir_src = osp.join(args.output_src, args.da, args.dset,
                                   names[args.s][0].upper() + names[args.t][0].upper())
    print(args.output_dir_src)
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    if not args.distill:
        print(args.output_dir_src + '/source_F.pt')
        args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_source_simp(args)

        args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
        for i in range(len(names)):
            if i == args.s:
                continue
            args.t = i
            args.name = names[args.s][0].upper() + names[args.t][0].upper()
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

            test_target_simp(args)

    if args.distill:
        for i in range(len(names)):
            if i == args.s:
                continue
            args.t = i
            args.name = names[args.s][0].upper() + names[args.t][0].upper()

            args.output_dir = osp.join(args.output, args.net_src + '_' + args.net, str(args.seed), args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
            if not osp.exists(args.output_dir):
                os.system('mkdir -p ' + args.output_dir)
            if not osp.exists(args.output_dir):
                os.mkdir(args.output_dir)

            args.out_file = open(osp.join(args.output_dir, 'log_tar.txt'), 'w')
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

            test_target_simp(args)
            copy_target_simp(args)