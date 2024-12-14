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
from data_list import ImageList
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth,our_CrossEntropy
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import torch.nn.functional as F
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.nn.functional import cosine_similarity

# def ssd(p, q):
#     return torch.sum((p - q) ** 2)


def custom_cosine_similarity_2d(x1, x2, eps=1e-8):
    # 计算点积，x1和x2的形状应该是 [batch_size, features]
    dot_product = torch.sum(x1 * x2, dim=1)

    # 计算两个向量的模
    norm_x1 = torch.sqrt(torch.sum(x1 ** 2, dim=1) + eps)
    norm_x2 = torch.sqrt(torch.sum(x2 ** 2, dim=1) + eps)

    # 计算余弦相似度，并调整其范围到[0, 1]
    cos_sim = (dot_product / (norm_x1 * norm_x2)).clamp(min=0, max=1)

    return cos_sim

def ssd(p, q):
    # 计算p和q之间的差值的平方
    squared_diff = (p - q) ** 2

    # 沿着batch维度（即第一个维度）对差值的平方求和，以计算每个batch的SSD
    ssd_values = torch.sum(squared_diff, dim=1)

    # 将SSD值转换为Python列表
    ssd_list = ssd_values.tolist()

    return ssd_list


def kl_divergence(p, q):
    # 避免分母为0
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    return torch.sum(p * torch.log(p / q))


def bhattacharyya_coefficient(p, q):
    return torch.sum(torch.sqrt(p * q))


def canberra_distance(p, q):
    return torch.sum(torch.abs(p - q) / (torch.abs(p) + torch.abs(q)))


def canberra_distance_per_batch(p, q):
    # 初始化一个空列表来存储每个batch的Canberra距离
    distances = []

    # 确保输入是批处理格式，即第一个维度是batch维度
    # 遍历每个batch
    for p_batch, q_batch in zip(p, q):
        # 计算Canberra距离
        numerator = torch.abs(p_batch - q_batch)
        denominator = torch.abs(p_batch) + torch.abs(q_batch)
        # 为了避免除以0的情况，使用clamp方法将分母中的最小值设置为一个非常小的正数
        distance = torch.sum(numerator / denominator.clamp(min=1e-10))

        # 将计算得到的距离添加到列表中
        distances.append(distance.item())  # 使用.item()将单个元素张量转换为Python数值

    return distances


def chebyshev_distance(p, q):
    return torch.max(torch.abs(p - q))


def chebyshev_distance_per_batch(p, q):
    # 初始化一个空列表来存储每个batch的Chebyshev距离
    distances = []

    # 遍历每个batch
    for p_batch, q_batch in zip(p, q):
        # 计算每个batch的Chebyshev距离
        # torch.max 返回的是一个元组(values, indices)，我们只需要values，所以用[0]
        distance = torch.max(torch.abs(p_batch - q_batch), dim=-1)[0]

        # 因为Chebyshev距离是基于最大差异的，我们需要再次使用torch.max找到最大的那个值
        max_distance = torch.max(distance)
        max_distance = 1 / (1 + max_distance)

        # 将计算得到的距离添加到列表中
        distances.append(max_distance.item())  # 使用.item()将单个元素张量转换为Python数值

    return distances


def kl_div_loss_per_batch(log_softmax_outputs, labels):
    # 初始化一个空列表来存储每个batch的损失
    losses = []

    # 遍历每个batch
    for log_softmax_output, label in zip(log_softmax_outputs, labels):
        # 计算当前batch的KL散度损失
        # 使用unsqueeze(0)是为了保持batch维度，因为kl_div期望的输入是batch维度的
        loss = F.kl_div(log_softmax_output.unsqueeze(0), label.unsqueeze(0), reduction='batchmean')

        # 将当前batch的损失添加到列表中
        losses.append(loss.item())  # 使用.item()将单个元素张量转换为Python数值

    return losses

def bhattacharyya_coefficient_stable(p, q, epsilon=1e-10):
    # 确保概率值非负且规范化
    p = torch.abs(p) / torch.clamp(torch.sum(p, dim=1, keepdim=True), min=epsilon)
    q = torch.abs(q) / torch.clamp(torch.sum(q, dim=1, keepdim=True), min=epsilon)

    # 应用阈值以避免下溢
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)

    # 计算每一对行之间的Bhattacharyya系数，并裁剪结果以确保其在[0, 1]范围内
    bc = torch.sum(torch.sqrt(p * q), dim=1)
    bc = torch.clamp(bc, min=0.0, max=1.0)

    return bc

# def cosine_similarity(p, q):
#     return torch.dot(p, q) / (torch.norm(p) * torch.norm(q))


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


def target_transform(x):
    # print(x)

    return x / x.sum()


def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    # txt_src = open(args.s_dset_path).readlines()
    # txt_test = open(args.test_dset_path).readlines()

    path = '/nfs/ofs-902-1/object-detection/zhujiankun/EDA/data/LDL/'
    tr_txt = open(path + f'{args.names[args.s]}_LDL/{args.names[args.s]}_ldl_train.txt').readlines()

    te_txt = open(path + f'{args.names[args.s]}_LDL/{args.names[args.s]}_ldl_test.txt').readlines()
    txt_test = open(path + f'{args.names[args.t]}_LDL/{args.names[args.t]}_ldl_test.txt').readlines()


    dsets["source_tr"] = ImageList(args,tr_txt, transform=image_train(),target_transform=target_transform)
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(args,te_txt, transform=image_test(),target_transform=target_transform)
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(args,txt_test, transform=image_test(),target_transform=target_transform)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    import torch.nn.functional as F
    start_test = True
    losses_kl = []
    losses_ssd = []
    # losses_kl = []
    losses_bc = []
    losses_cos = []
    losses_canb = []
    losses_che = []
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            # print(i)

            data = iter_test.next()
            inputs = data[0]
            labels = data[1].cuda()

            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            softmax_outputs = torch.softmax(outputs,dim=1)
            log_softmax_outputs = torch.log(outputs.softmax(-1))

            loss_kl = kl_div_loss_per_batch(log_softmax_outputs, labels)
            # loss = F.kl_div(log_softmax_outputs, labels, reduction='batchmean')

            loss_ssd = ssd(softmax_outputs, labels)
            # print(loss_ssd)
            # loss_kl = kl_divergence(log_softmax_outputs, labels)
            # print('softmax_outputs', softmax_outputs.shape)
            # print('labels', labels.shape)
            loss_bc = bhattacharyya_coefficient_stable(softmax_outputs, labels)
            # print(loss_bc)
            loss_cos = custom_cosine_similarity_2d(outputs, labels).mean()
            loss_canb = canberra_distance_per_batch(softmax_outputs, labels)
            loss_che = chebyshev_distance_per_batch(outputs, labels)

            # losses_kl.extend(loss_kl)
            losses_ssd.extend(loss_ssd)
            losses_kl.extend(loss_kl)
            losses_bc.extend(loss_bc)
            # print(len(losses_bc))
            losses_cos.append(loss_cos)
            losses_canb.extend(loss_canb)
            losses_che.extend(loss_che)


    accuracy = sum(losses_kl) / len(losses_kl)
    acc_ssd = sum(losses_ssd) / len(losses_ssd)
    acc_bc = sum(losses_bc) / len(losses_bc)
    acc_canb = sum(losses_canb) / len(losses_canb)
    acc_che = sum(losses_che) / len(losses_che)
    acc_cos = sum(losses_cos) / len(losses_cos)
    print('acc_ssd:', acc_ssd)
    print('acc_kl:', accuracy)
    # print('acc_kl:', acc_kl)
    print('acc_bc:', acc_bc)
    print('acc_canb:', acc_canb)
    print('acc_che:', acc_che)
    print('acc_cos:', acc_cos)


    return accuracy * 100

def cal_acc_oda(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
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

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()
    initc = np.array([[0], [1]])
    kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1,1))
    threshold = (kmeans.cluster_centers_).mean()

    predict[ent>threshold] = args.class_num
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int),:]

    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()

    return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    # return np.mean(acc), np.mean(acc[:-1])

def train_source(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

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
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = sys.maxsize
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 20
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    # count = 0
    while iter_num < max_iter:
        # count += 1
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()
        # print(labels_source)
        # raise Exception
        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        # print(inputs_source.shape)
        # print(labels_source)
        # raise Exception
        outputs_source = netC(netB(netF(inputs_source)))
        outputs_source = F.log_softmax(outputs_source, dim=1)
        classifier_loss = F.kl_div(outputs_source.double(), labels_source.double(), reduction='batchmean')

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            print('classifier_loss', classifier_loss)
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}'.format(args.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te = cal_acc(dset_loaders['source_te'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc_s_te <= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()
                torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
                torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
                torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

            netF.train()
            netB.train()
            netC.train()
    return netF, netB, netC


def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

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

    if args.da == 'oda':
        acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB, netC)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name, acc_os2, acc_os1, acc_unknown)
    else:
        if args.dset=='VISDA-C':
            acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n' + acc_list
        else:
            acc = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=100, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=16, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='Emo_Rec', choices=['VISDA-C', 'office', 'office-home', 'Emo_Rec', 'Emo_ldl'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    args = parser.parse_args()


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
    if args.dset == 'Emo_ldl':
        names = ['Twitter','Flickr']
        args.names = names
        args.class_num = 8

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    # folder = '../list_file/'
    args.s_dset_path = f'/nfs/ofs-902-1/object-detection/zhujiankun/EDA/data/LDL/{names[args.s]}_LDL/{names[args.s]}_ldl.txt'
    args.test_dset_path = f'/nfs/ofs-902-1/object-detection/zhujiankun/EDA/data/LDL/{names[args.t]}_LDL/{names[args.t]}_ldl.txt'

    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]

    args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper())
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_source(args)

    args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        # folder = '../list_file/'
        # args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        # args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]
            if args.da == 'oda':
                args.class_num = 25
                args.src_classes = [i for i in range(25)]
                args.tar_classes = [i for i in range(65)]

        test_target(args)