import argparse
from collections import OrderedDict
import json
import math
import os
import sys
import time
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from data import get_data
from tokenizer import SimpleTokenizer
import models
import losses
import utils
from utils import AverageMeter, ProgressMeter, recall_at_k, get_gpu_info


def get_args_parser():
    parser = argparse.ArgumentParser(description='Meme_Text_Retrieval_CLIP training', add_help=False, )
    parser.add_argument(
        "--train-data",
        type=str,
        default='../data/meme_retrieval_data/training_set.json',
        help="Path to training data.",
    )
    parser.add_argument(
        '--val-data',
        type=str,
        default='../data/meme_retrieval_data/validation_set.json',
        help='Path to validation data.'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        default='../data/memecap/meme-cap-main/data/memes-test.json',
        help='Path to test data.'
    )
    parser.add_argument(
        '--root',
        type=str,
        default='../data/',
        help='Root directory of images.'
    )
    # candidates: ['merge', 'extend', 'first']
    parser.add_argument('--caption-preprocess', default='first', type=str)
    # list of filenames for augmented captions
    # parser.add_argument('--augmented_caption_filelist', nargs='+', help='list of augmented caption filenames, seperated by space')
    # parser.add_argument('--aug-text', action='store_true', help='set to True for LaCLIP')

    parser.add_argument('--image-root', default='../data/meme_retrieval_data/meme_images', 
                        type=str, help='path to image dataset')
    parser.add_argument('--output-dir', default='./output', type=str, help='output dir')

    parser.add_argument('--model', default='CLIP_VITL14_336', type=str)
    parser.add_argument('--resume', default='', type=str, help='path to resume from')

    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int) # Or steps?
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=16, type=int,
                        help='number of samples per-gpu') # To be decided
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=150, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')

    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--distributed', default=False, type=bool,
                        help='distributed training')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    return parser


best_r_at_1 = 0


def main(args):
    utils.init_distributed_mode(args)

    global best_r_at_1

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create model
    print("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)()
    model.cuda(args.gpu)
    # print(model)
    print('Distributed:', args.distributed)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200)

    # define loss function (criterion) and optimizer
    criterion = losses.CLIPLoss().cuda(args.gpu)

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            best_r_at_1 = checkpoint['best_r_at_1']
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_r_at_1 = latest_checkpoint['best_r_at_1']
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    cudnn.benchmark = True

    # Data loading code
    print("=> creating dataset")
    tokenizer = SimpleTokenizer()
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # train_transform = transforms.Compose([
    #         transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    #         transforms.ToTensor(),
    #         normalize
    #     ])
    # val_transform = transforms.Compose([
    #         transforms.Resize(224),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize
    #     ])
    train_transform = utils.transform(336)
    val_transform = utils.transform(336)
    # val_dataset = datasets.ImageFolder(os.path.join(args.image_root, 'val'), transform=val_transform)


    # if args.distributed:
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    # else:
    #     val_sampler = None

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)

    data = get_data(args, (train_transform, val_transform), tokenizer=tokenizer)
    print('dataset size: %d' % data['train'].dataloader.num_samples)
    train_loader = data['train'].dataloader
    val_loader = data['val'].dataloader
    loader_len = train_loader.num_batches

    lr_schedule = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
                                         loader_len // args.update_freq,
                                         warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)
    print('lr_schedule:', len(lr_schedule), lr_schedule)
    if utils.is_main_process() and args.output_dir is not None:
        args.log_dir = os.path.join(args.output_dir, 'tb_logs')
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    print(args)
    print("=> test before training:")
    print('=> validation set: ')
    print(validate(val_loader, model, tokenizer, args))
    print('=> test set: ')
    print(validate(data['test'].dataloader, model, tokenizer, args))

    print("=> beginning training")
    for epoch in range(args.start_epoch, args.epochs):
        get_gpu_info()
        if args.distributed:
            data['train'].set_epoch(epoch)
        train_loader = data['train'].dataloader

        # train for one epoch
        train_stats = train(train_loader, log_writer, model, criterion, optimizer, scaler, epoch, lr_schedule, args)
        
        print('=> validation: ')
        val_stats = validate(val_loader, model, tokenizer, args)
        print('=> test: ')
        print(validate(data['test'].dataloader, model, tokenizer, args))
        acc1 = val_stats['r@1']

        is_best = acc1 > best_r_at_1
        best_r_at_1 = max(acc1, best_r_at_1)
        print("=> saving checkpoint")
        utils.save_on_master({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_r_at_1': best_r_at_1,
                'args': args,
            }, is_best, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}

        # log test stats to log_writer (tensorboard)
        if log_writer is not None:
            for k, v in log_stats.items():
                if k.startswith('test'):
                    log_writer.add_scalar(k, v, epoch)

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
    
    print("=> Training finished")
    # print("Test the model after fine-tuning: ")
    # print(validate(data['test'].dataloader, model, tokenizer, args))


def train(train_loader, log_writer, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['loss', 'clip_loss', 'clip_acc']

    loader_len = train_loader.num_batches
    iters_per_epoch = loader_len // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it] 

        inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]

        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(*inputs)
            # print(outputs)
            loss_dict = criterion(outputs)
            loss = loss_dict['loss']
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # clamp logit scale to [0, 100]
        # print('Before:', utils.get_model(model).logit_scale)
        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = utils.get_model(model).logit_scale.exp().item()
        # print('After:', utils.get_model(model).logit_scale)
        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        # save to log_writer (tensorboard)
        if log_writer is not None:
            for k, v in loss_dict.items():
                log_writer.add_scalar(k, v.item(), it)
            log_writer.add_scalar('scaler', scaler.get_scale(), it)
            log_writer.add_scalar('logit', logit_scale, it)
            log_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], it)

        if optim_iter % args.print_freq == 0:
            progress.display(optim_iter)

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


def validate(val_loader, model, tokenizer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    T2I_top1 = AverageMeter('T2I R@1', ':6.2f')
    T2I_top5 = AverageMeter('T2I R@5', ':6.2f')
    T2I_top10 = AverageMeter('T2I R@10', ':6.2f')
    
    I2T_top1 = AverageMeter('I2T R@1', ':6.2f')
    I2T_top5 = AverageMeter('I2T R@5', ':6.2f')
    I2T_top10 = AverageMeter('I2T R@10', ':6.2f')
    # top10 = AverageMeter('Acc@10', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, T2I_top1, T2I_top5, T2I_top10, I2T_top1, I2T_top5, I2T_top10],
        prefix='Validation: ')

    # switch to evaluate mode
    model.eval()

    print('=> encoding captions')

    with torch.no_grad():
        text_features_all = []
        image_features_all = []
        end = time.time()
        
        for i, (images, texts) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            texts = texts.cuda(args.gpu, non_blocking=True)
            
            # encode texts
            # texts_ids = tokenizer(texts).cuda(args.gpu, non_blocking=True)
            text_features = utils.get_model(model).encode_text(texts)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_all.append(text_features.cpu())
            # encode images
            image_features = utils.get_model(model).encode_image(images)
            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features_all.append(image_features.cpu())

        print(len(image_features_all), len(text_features_all))
        # cosine similarity as logits
        # Convert list of tensors to a single tensor
        image_features_all = torch.cat(image_features_all, 0).type(torch.float32)
        print(image_features_all.shape)
        image_features_all = image_features_all / image_features_all.norm(p=2, dim=-1, keepdim=True)

        # Convert list of tensors to a single tensor
        text_features_all = torch.cat(text_features_all, 0).type(torch.float32)
        print(text_features_all.shape)
        text_features_all = text_features_all / text_features_all.norm(p=2, dim=-1, keepdim=True)

        logits_per_image = image_features_all @ text_features_all.t()

        # # measure accuracy and record loss
        # acc1, acc5 = accuracy(logits_per_image, target, topk=(1, 5))
        img2text_recall_at_k = recall_at_k(logits_per_image, prefix = 'i2t_')
        text2img_recall_at_k = recall_at_k(logits_per_image.T, prefix = 't2i_')
        
        T2I_top1.update(text2img_recall_at_k['t2i_r1'], 1)
        T2I_top5.update(text2img_recall_at_k['t2i_r5'], 1)
        T2I_top10.update(text2img_recall_at_k['t2i_r10'], 1)
        I2T_top1.update(img2text_recall_at_k['i2t_r1'], 1)
        I2T_top5.update(img2text_recall_at_k['i2t_r5'], 1)
        I2T_top10.update(img2text_recall_at_k['i2t_r10'], 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print(img2text_recall_at_k)
        print(text2img_recall_at_k)
        if i % args.print_freq == 0:
            progress.display(i)

    progress.synchronize()
    print('The mean img2text R@K: {}; The mean text2img R@K: {}'
          .format(img2text_recall_at_k['i2t_r_mean'], text2img_recall_at_k['t2i_r_mean']))
    return {'r@1': (img2text_recall_at_k['i2t_r_mean'] + text2img_recall_at_k['t2i_r_mean']) / 2}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme_Text_Retrieval_CLIP training', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)