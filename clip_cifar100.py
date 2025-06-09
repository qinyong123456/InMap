import argparse
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import clip
from torch.utils.data import Subset
import os
import random

# CIFAR100类别名称
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

# 简单文本模板
cifar100_single_template = [
    'a photo of a {}.',
]

# 多样化文本模板
cifar100_7_templates = [
    'itap of a {}.',
    'a origami {}.',
    'a bad photo of the {}.',
    'a photo of the large {}.',
    'a {} in a video game.',
    'art of the {}.',
    'a photo of the small {}.',
]

def get_cifar100_datasets(data_path, val_size=0.2, seed=42):
    """加载CIFAR100数据集并划分为训练集、验证集和测试集"""
    # 定义预处理（调整大小以匹配CLIP模型输入）
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711))  # CLIP的归一化参数
    ])
    
    # 加载完整训练集
    train_dataset = datasets.CIFAR100(
        root=data_path, train=True, download=True, transform=preprocess
    )
    
    # 划分训练集和验证集
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(val_size * num_train)
    
    # 设置随机种子确保划分可复现
    torch.manual_seed(seed)
    random.seed(seed)
    
    # 随机打乱索引
    random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    
    # 创建子集
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    
    # 加载测试集
    test_dataset = datasets.CIFAR100(
        root=data_path, train=False, download=True, transform=preprocess
    )
    
    return train_subset, val_subset, test_dataset

def zeroshot_classifier(clip, model, classnames, templates):
    """创建零样本分类器权重"""
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def image_opt(feat, init_classifier, plabel, lr=10, iter=2000, tau_i=0.04, alpha=0.6):
    """优化视觉代理分类器"""
    ins, dim = feat.shape
    val, idx = torch.max(plabel, dim=1)
    mask = val > alpha
    plabel[mask, :] = 0
    plabel[mask, idx[mask]] = 1
    base = feat.T @ plabel
    classifier = init_classifier.clone()
    pre_norm = float('inf')
    for i in range(0, iter):
        prob = F.softmax(feat @ classifier / tau_i, dim=1)
        grad = feat.T @ prob - base
        temp = torch.norm(grad)
        if temp > pre_norm:
            lr /= 2.
        pre_norm = temp
        classifier -= (lr / (ins * tau_i)) * grad
        classifier = F.normalize(classifier, dim=0)
    return classifier

def sinkhorn(M, tau_t=0.01, gamma=0, iter=20):
    """使用Sinkhorn算法优化伪标签分布"""
    row, col = M.shape
    P = F.softmax(M / tau_t, dim=1)
    P /= row
    if gamma > 0:
        q = torch.sum(P, dim=0, keepdim=True)
        q = q**gamma
        q /= torch.sum(q)
    for it in range(0, iter):
        # 列归一化
        P /= torch.sum(P, dim=0, keepdim=True)
        if gamma > 0:
            P *= q
        else:
            P /= col
        # 行归一化
        P /= torch.sum(P, dim=1, keepdim=True)
        P /= row
    P *= row  # 保持每行和为1
    return P

def accuracy(output, target, topk=(1,)):
    """计算准确率"""
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def main():
    parser = argparse.ArgumentParser(description='CLIP for CIFAR100')
    parser.add_argument('--data_path', default='./data', type=str, help='dataset path')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50',
                        choices=['RN50', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'],
                        help='model architecture')
    parser.add_argument('-j', '--workers', default=4, type=int, help='data loading workers')
    parser.add_argument('--iters_proxy', default=2000, type=int, help='iterations for vision proxy')
    parser.add_argument('--iters_sinkhorn', default=20, type=int, help='iterations for Sinkhorn')
    parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=10, type=float, help='learning rate')
    parser.add_argument('--tau_t', default=0.01, type=float, help='temperature for text')
    parser.add_argument('--tau_i', default=0.04, type=float, help='temperature for image')
    parser.add_argument('--alpha', default=0.6, type=float, help='threshold for confident samples')
    parser.add_argument('--gamma', default=0.0, type=float, help='weight for column constraint')
    args = parser.parse_args()
    print(args)

    # 加载CLIP模型
    print('Loading CLIP model...')
    model, _ = clip.load(args.arch)
    model = model.cuda()
    model.eval()

    # 加载CIFAR100数据集
    print('Loading CIFAR100 dataset...')
    train_set, val_set, test_set = get_cifar100_datasets(args.data_path)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    # 提取训练集图像特征
    print('Extracting training image features...')
    with torch.no_grad():
        train_image_feat = []
        train_image_label = []
        for images, target in train_loader:
            images = images.cuda()
            target = target.cuda()
            image_features = model.encode_image(images)
            train_image_feat.append(F.normalize(image_features, dim=1))
            train_image_label.append(target)
    train_image_feat = torch.cat(train_image_feat, dim=0)
    train_image_label = torch.cat(train_image_label, dim=0)
    print(f"Training features shape: {train_image_feat.shape}")

    # 提取测试集图像特征
    print('Extracting test image features...')
    with torch.no_grad():
        test_image_feat = []
        test_image_label = []
        for images, target in test_loader:
            images = images.cuda()
            target = target.cuda()
            image_features = model.encode_image(images)
            test_image_feat.append(F.normalize(image_features, dim=1))
            test_image_label.append(target)
    test_image_feat = torch.cat(test_image_feat, dim=0)
    test_image_label = torch.cat(test_image_label, dim=0)
    print(f"Test features shape: {test_image_feat.shape}")

    # 确保所有特征和分类器使用相同的数据类型
    dtype = train_image_feat.dtype

    # 创建零样本分类器
    print('Creating zero-shot classifier...')
    text_classifier = zeroshot_classifier(clip, model, cifar100_classes, cifar100_7_templates)
    text_classifier = text_classifier.to(dtype)
    
    # 评估零样本分类器性能
    test_image_feat = test_image_feat.to(dtype)
    logits_t = test_image_feat @ text_classifier
    acc1, acc5 = accuracy(logits_t, test_image_label, topk=(1, 5))
    top1 = (acc1 / len(test_image_label)) * 100
    print(f"Zero-shot accuracy: {top1:.2f}%")

    # 无Sinkhorn的视觉代理
    print('Optimizing vision proxy without Sinkhorn...')
    # 使用训练集特征计算伪标签
    train_logits_t = train_image_feat @ text_classifier
    plabel = F.softmax(train_logits_t / args.tau_t, dim=1)
    
    # 修复维度匹配问题：确保plabel的类别数与CIFAR100一致
    assert plabel.shape[1] == len(cifar100_classes), f"plabel shape: {plabel.shape}, classes: {len(cifar100_classes)}"
    
    image_classifier = image_opt(
        train_image_feat, text_classifier, plabel, 
        lr=args.lr, iter=args.iters_proxy, tau_i=args.tau_i, alpha=args.alpha
    )
    image_classifier = image_classifier.to(dtype)
    logits_i = test_image_feat @ image_classifier
    acc1, acc5 = accuracy(logits_i, test_image_label, topk=(1, 5))
    top1 = (acc1 / len(test_image_label)) * 100
    print(f"Vision proxy accuracy (without Sinkhorn): {top1:.2f}%")

    # 使用Sinkhorn的视觉代理
    print('Optimizing vision proxy with Sinkhorn...')
    # 使用训练集特征生成Sinkhorn优化的伪标签
    plabel = sinkhorn(train_logits_t, args.tau_t, args.gamma, args.iters_sinkhorn)
    
    image_classifier = image_opt(
        train_image_feat, text_classifier, plabel, 
        lr=args.lr, iter=args.iters_proxy, tau_i=args.tau_i, alpha=args.alpha
    )
    image_classifier = image_classifier.to(dtype)
    logits_i = test_image_feat @ image_classifier
    acc1, acc5 = accuracy(logits_i, test_image_label, topk=(1, 5))
    top1 = (acc1 / len(test_image_label)) * 100
    print(f"Vision proxy accuracy (with Sinkhorn): {top1:.2f}%")

if __name__ == '__main__':
    main()    
