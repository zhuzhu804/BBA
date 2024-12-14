import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from loss import KnowledgeDistillationLoss
from Modules.masking import Masking
import torch.nn.functional as f

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
    return transforms.Compose([
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
    return transforms.Compose([
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

    # print(len(open(args.te_txt).readlines()))
    dsets["test"] = ImageList_idx(open(args.te_txt).readlines(), transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False,
                                      num_workers=args.worker, drop_last=False)
    # dsets = {}
    # dset_loaders = {}
    # train_bs = args.batch_size
    # txt_tar = open(args.t_dset_path).readlines()
    # txt_test = open(args.test_dset_path).readlines()
    #
    # if not args.da == 'uda':
    #     label_map_s = {}
    #     for i in range(len(args.src_classes)):
    #         label_map_s[args.src_classes[i]] = i
    #
    #     new_tar = []
    #     for i in range(len(txt_tar)):
    #         rec = txt_tar[i]
    #         reci = rec.strip().split(' ')
    #         if int(reci[1]) in args.tar_classes:
    #             if int(reci[1]) in args.src_classes:
    #                 line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
    #                 new_tar.append(line)
    #             else:
    #                 line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
    #                 new_tar.append(line)
    #     txt_tar = new_tar.copy()
    #     txt_test = txt_tar.copy()
    #
    # dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    # dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
    #                                     drop_last=False)
    # dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    # dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
    #                                   drop_last=False)

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
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
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
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    elif args.net == 'vit':
        netF = network.ViT().cuda()
    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath), strict=False)
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False
    ### add teacher module
    if args.net[0:3] == 'res':
        netF_t = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF_t = network.VGGBase(vgg_name=args.net).cuda()
    elif args.net == 'vit':
        netF_t = network.ViT().cuda()
    netB_t = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    ### initial from student
    netF_t.load_state_dict(netF.state_dict())
    netB_t.load_state_dict(netB.state_dict())

    ### remove grad
    for k, v in netF_t.named_parameters():
        v.requires_grad = False
    for k, v in netB_t.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    # print(len(dset_loaders["target"]))
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    masking = Masking(
        block_size=args.mask_block_size,
        ratio=args.mask_ratio,
        color_jitter_s=args.mask_color_jitter_s,
        color_jitter_p=args.mask_color_jitter_p,
        blur=args.mask_blur,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))

    acc_init = 0
    acc_init_t = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            netF_t.eval()
            netB_t.eval()
            mem_label, dd, acc_s_te_t = obtain_label(dset_loaders['test'], netF_t, netB_t, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            dd = torch.from_numpy(dd).cuda()

            netF.train()
            netB.train()


            if acc_s_te_t >= acc_init_t:
                acc_init_t = acc_s_te_t
                best_netF = netF_t.state_dict()
                best_netB = netB_t.state_dict()
                best_netC = netC.state_dict()
                if args.issave:
                    torch.save(best_netF, osp.join(args.output_dir_t, "source_F.pt"))
                    torch.save(best_netB, osp.join(args.output_dir_t, "source_B.pt"))
                    torch.save(best_netC, osp.join(args.output_dir_t, "source_C.pt"))

        inputs_test = inputs_test.cuda()
        inputs_test_masked = masking(inputs_test)
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        features_test_masked = netB(netF(inputs_test_masked))
        outputs_test_masked = netC(features_test_masked)

        if args.cls_par > 0:
            # print('1')
            pred = mem_label[tar_idx]
            pred_soft = dd[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            # classifier_loss = PCCEVE8(lambda_0=0.5)(outputs_test, pred)
            classifier_loss *= args.cls_par
            masking_loss_value = nn.CrossEntropyLoss()(outputs_test_masked, pred)
            # masking_loss_value = PCCEVE8(lambda_0=0.5)(outputs_test_masked, pred)
            masking_loss_value *= args.cls_par
            classifier_loss += masking_loss_value
            if args.kd:
                kd_loss = KnowledgeDistillationLoss()(outputs_test, pred_soft)
                classifier_loss += kd_loss
                kd_loss_masked = KnowledgeDistillationLoss()(outputs_test_masked, pred_soft)
                classifier_loss += kd_loss_masked
                # kd_loss_cross = KnowledgeDistillationLoss()(outputs_test_masked, outputs_test)
                # classifier_loss += kd_loss_cross


            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        # EMA update for the teacher
        with torch.no_grad():
            m = 0.001  # momentum parameter
            for param_q, param_k in zip(netF.parameters(), netF_t.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(netB.parameters(), netB_t.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if args.dset == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter,
                                                                            acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()
                if args.issave:
                    torch.save(best_netF, osp.join(args.output_dir, "source_F.pt"))
                    torch.save(best_netB, osp.join(args.output_dir, "source_B.pt"))
                    torch.save(best_netC, osp.join(args.output_dir, "source_C.pt"))

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()

    # TODO
    # if args.issave:
    #     torch.save(best_netF, osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
    #     torch.save(best_netB, osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
    #     torch.save(best_netC, osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))



    return netF, netB, netC


class PCCEVE8(nn.Module):
    """
    0 amusement
    1 anger
    2 awe
    3 contentment
    4 disgust
    5 excitement
    6 fear
    7 sadness
    Positive: amusement, awe, contentment, excitement
    Negative: anger, disgust, fear, sadness
    """

    def __init__(self, lambda_0):
        super(PCCEVE8, self).__init__()
        self.POSITIVE = {0, 2, 3, 5}
        self.NEGATIVE = {1, 4, 6, 7}

        self.lambda_0 = lambda_0

        self.f0 = nn.CrossEntropyLoss(reduce=False)

    def forward(self, y_pred, y):
        batch_size = y_pred.size(0)
        weight = [1] * batch_size

        out = self.f0(y_pred, y)
        _, y_pred_label = f.softmax(y_pred, dim=1).topk(k=1, dim=1)
        y_pred_label = y_pred_label.squeeze(dim=1)
        y_numpy = y.cpu().numpy()
        y_pred_label_numpy = y_pred_label.cpu().numpy()
        for i, y_numpy_i, y_pred_label_numpy_i in zip(range(batch_size), y_numpy, y_pred_label_numpy):
            if (y_numpy_i in self.POSITIVE and y_pred_label_numpy_i in self.NEGATIVE) or (
                    y_numpy_i in self.NEGATIVE and y_pred_label_numpy_i in self.POSITIVE):
                weight[i] += self.lambda_0
        weight_tensor = torch.from_numpy(np.array(weight)).cuda()
        out = out.mul(weight_tensor)
        out = torch.mean(out)

        return out


def map_classes_to_binary(classes):
    # print(classes)
    return [0 if x in [0, 2, 3, 5] else 1 for x in classes]


def fused_distance(fea, centers):
    # 计算欧氏距离
    dist_euclidean = cdist(fea, centers, 'euclidean')

    # 计算余弦距离
    norm_fea = fea / np.linalg.norm(fea, axis=1, keepdims=True)
    norm_centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    dist_cosine = 1 - np.dot(norm_fea, norm_centers.T)

    # 计算曼哈顿距离
    dist_manhattan = cdist(fea, centers, 'cityblock')

    # 融合距离度量
    dist_fused = dist_euclidean + dist_cosine + dist_manhattan
    return dist_fused


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label_fusion(loader, netF, netB, netC, args):
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
    print('accuracy', accuracy)
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

    # 迭代更新聚类中心
    for round in range(5):
        # 使用融合的距离度量计算所有点到各聚类中心的距离
        dd = fused_distance(all_fea, initc)

        # 为每个点分配最近的聚类中心
        pred_label = dd.argmin(axis=1)

        # 更新聚类中心
        for i in range(len(initc)):
            if (pred_label == i).any():  # 避免除以零
                initc[i] = all_fea[pred_label == i].mean(axis=0)

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

    return pred_label.astype('int'), dd



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
    # dd = cdist(all_fea, initc, args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        # dd = cdist(all_fea, initc, args.distance)
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

    return pred_label.astype('int'), dd, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=1, help="source")
    parser.add_argument('--t', type=int, default=0, help="target")
    parser.add_argument('--max_epoch', type=int, default=100, help="max iterations")
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=0, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home-luban',
                        choices=['VISDA-C', 'office', 'office-home-luban', 'office-caltech', 'Emo_Rec', 'Art_FI', 'E6_FI'])
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--kd', type=bool, default=True)
    parser.add_argument('--se', type=bool, default=False)
    parser.add_argument('--nl', type=bool, default=False)

    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_t', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--mask_block_size', default=32, type=int)
    parser.add_argument('--mask_ratio', default=0.5, type=float)
    parser.add_argument('--mask_color_jitter_s', default=0, type=float)
    parser.add_argument('--mask_color_jitter_p', default=0, type=float)
    parser.add_argument('--mask_blur', default=False, type=bool)
    args = parser.parse_args()
    print('lr',args.lr)
    print('s', args.s)
    print('t', args.t)
    if args.dset == 'office-home-luban':
        names = ['Art', 'Clipart']
        args.class_num = 65
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'Emo_Rec':
        names = ['FI', 'EmoSet']
        args.class_num = 8
    if args.dset == 'Art_FI':
        names = ['FI','Art']
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

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        #
        # folder = '/nfs/ofs-902-1/object-detection/zhujiankun/EDA/code/SHOT/list_file/Art_FI'
        # args.tr_txt = os.path.join(folder, f'FI_test_8.txt')
        # args.te_txt = os.path.join(folder, f'FI_test_8.txt')
        folder = '/home/cx_xchen/zhuzhu/apps/Emo-SFDA/BBA/list_file/Emotion6_FI'
        args.tr_txt = os.path.join(folder, f'{names[args.t]}_test.txt')
        args.te_txt = os.path.join(folder, f'{names[args.t]}_test.txt')


        if args.dset == 'office-home-luban':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() + names[args.t][0].upper())
        args.output_dir_t = osp.join(args.output_t, args.da, args.dset, names[args.s][0].upper() + names[args.t][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        if not osp.exists(args.output_dir_t):
            os.system('mkdir -p ' + args.output_dir_t)
        if not osp.exists(args.output_dir_t):
            os.mkdir(args.output_dir_t)

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)