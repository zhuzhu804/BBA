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
torch.autograd.set_detect_anomaly(True)
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


def ssd(p, q):
    # 计算p和q之间的差值的平方
    squared_diff = (p - q) ** 2

    # 沿着batch维度（即第一个维度）对差值的平方求和，以计算每个batch的SSD
    ssd_values = torch.sum(squared_diff, dim=1)

    # 将SSD值转换为Python列表
    ssd_list = ssd_values.tolist()

    return ssd_list


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


def custom_cosine_similarity_2d(x1, x2, eps=1e-8):
    # 计算点积，x1和x2的形状应该是 [batch_size, features]
    dot_product = torch.sum(x1 * x2, dim=1)

    # 计算两个向量的模
    norm_x1 = torch.sqrt(torch.sum(x1 ** 2, dim=1) + eps)
    norm_x2 = torch.sqrt(torch.sum(x2 ** 2, dim=1) + eps)

    # 计算余弦相似度，并调整其范围到[0, 1]
    cos_sim = (dot_product / (norm_x1 * norm_x2)).clamp(min=0, max=1)

    return cos_sim


def target_transform(x):
    # print(x)

    return x / x.sum()


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size

    path = '/nfs/ofs-902-1/object-detection/zhujiankun/EDA/data/LDL/'
    txt_test = open(path + f'{args.names[args.t]}_LDL/{args.names[args.t]}_ldl_test.txt').readlines()

    dsets["target"] = ImageList(args,txt_test, transform=image_train(),target_transform=target_transform)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=False)

    dsets["test"] = ImageList(args,txt_test, transform=image_test(),target_transform=target_transform)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False,
                                      num_workers=args.worker, drop_last=False)

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

    acc_init = sys.maxsize

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()


        if inputs_test.size(0) == 1:
            continue
        if iter_num == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            netF_t.eval()
            netB_t.eval()
            mem_label = obtain_label(dset_loaders['test'], netF_t, netB_t, netC, args).cuda()
            # mem_label = torch.from_numpy(mem_label).cuda()
            # dd = torch.from_numpy(dd).cuda()

            netF.train()
            netB.train()

        inputs_test = inputs_test.cuda()
        inputs_test_masked = masking(inputs_test)
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        outputs_test = torch.nn.Softmax(dim=1)(outputs_test)
        features_test_masked = netB(netF(inputs_test_masked))
        outputs_test_masked = netC(features_test_masked)
        outputs_test_masked = torch.nn.Softmax(dim=1)(outputs_test_masked)
        if args.cls_par > 0:
            # print(tar_idx)
            # print(len(all_output))
            all_output = mem_label[tar_idx, :]
            if args.kd:
                classifier_loss = nn.KLDivLoss(reduction='batchmean')(outputs_test.log(), all_output)
                kd_loss_masked = nn.KLDivLoss(reduction='batchmean')(outputs_test_masked.log(), all_output)
                classifier_loss += kd_loss_masked

            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()
        if args.ent:
            entropy_loss = torch.mean(loss.Entropy(outputs_test))
            if args.gent:
                msoftmax = outputs_test.mean(dim=0)
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
                acc_s_te = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                # log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter,
                #                                                             acc_s_te) + '\n' + acc_list
            else:
                acc_s_te = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            if acc_s_te <= acc_init:
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



def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label(loader, netF, netB, netC, args):
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.metrics import classification_report
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                # print('all_output',all_output.shape)
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                # print('all_output',all_output.shape)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    return all_output


def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()


    acc = cal_acc(dset_loaders['test'], netF, netB, netC, False)
    log_str = '\nTask: {}, Accuracy = {:.2f}%'.format(args.name, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=1000, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home-luban',
                        choices=['VISDA-C', 'office', 'office-home-luban', 'office-caltech','Emo_Rec','Emo_ldl'])
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
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

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        # folder = './data/'
        # args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        # args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        # args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        # folder = '/nfs/ofs-902-1/object-detection/zhujiankun/EDA/code/SHOT/list_file/Emo_Rec/image_full_list'
        # args.tr_txt = os.path.join(folder, f'{names[args.t]}_test.txt')
        # args.te_txt = os.path.join(folder, f'{names[args.t]}_test.txt')


        if args.dset == 'office-home-luban':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() + names[args.t][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        test_target(args)
        train_target(args)