import argparse
from collections import OrderedDict
import json
import os
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


from data import MemecapDataset, get_dataset
from tokenizer import SimpleTokenizer
import models
import utils
from utils import AverageMeter, ProgressMeter, recall_at_k, get_gpu_info
from clip import clip

def get_args_parser():
    parser = argparse.ArgumentParser(description='Meme_Text_Retrieval_CLIP testing', add_help=False, )
    parser.add_argument(
        '--test-data',
        type=str,
        default='/data/memecap/meme-cap-main/data/memes-test-ocr.json',
        help='Path to test data.'
    )
    parser.add_argument(
        '--root',
        type=str,
        default='/home/bjc154/meme_text_retrieval_p1',
        help='Root directory of images.'
    )
    parser.add_argument('--caption-preprocess', default='first', type=str)
    parser.add_argument('--image-root', default='/home/bjc154/meme_text_retrieval_p1/data/meme_retrieval_data/meme_images', 
                        type=str, help='path to image dataset')
    parser.add_argument('--output-dir', default='./output/tb_logs/18.10.2024_EP20_lr_1e-5/', type=str, help='output dir')
    parser.add_argument('--checkpoint', default='checkpoint_best.pt', type=str)
    parser.add_argument('--text_type', default='ocr', type=str) # img_captions, meme_captions, title, OCR 

    parser.add_argument('--model', default='CLIP_VITL14_336', type=str)

    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--distributed', default=False, type=bool,
                        help='distributed training')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='number of samples per-gpu') # To be decided
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    return parser



def main(args):
    # create model
    print("=> creating model: {}".format(args.model))
    device = torch.device('cuda' if torch.cuda.is_available() else  'cpu')
    print(device)
    # model, preprocess = clip.load("ViT-B/16", device=device)
    # val_transform = utils.transform(224)
    
    # auto-resume from latest checkpoint in output directory
    latest = os.path.join(args.output_dir, args.checkpoint)
    if os.path.isfile(latest):
        print("=> loading latest checkpoint '{}'".format(latest))
        model = getattr(models, args.model)()
        latest_checkpoint = torch.load(latest, map_location='cpu')
        new_state = {}
        for key in latest_checkpoint['state_dict']:
            new_key = key.replace("module.", "")
            new_state[new_key] = latest_checkpoint['state_dict'][key]
        latest_checkpoint['state_dict'] = new_state
        model.load_state_dict(latest_checkpoint['state_dict'])
    else:
        model = getattr(models, args.model)()
    model.cuda(args.gpu)
    val_transform = utils.transform(336)
    cudnn.benchmark = True
    # Data loading code
    print("=> creating dataset")
    tokenizer = SimpleTokenizer()
    
    test_dataset = MemecapDataset(
        args.test_data,
        val_transform,
        text_type=args.text_type,
        caption_preprocess = args.caption_preprocess,
        root=args.root,
        tokenizer=tokenizer
    )
    data = {"test": get_dataset(args, test_dataset, is_train=False)}

    print(args)
    print('=> test set: ')
    print(validate(data['test'].dataloader, model, tokenizer, args))
    print("=> Finished test \n\n\n\n")

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

        print(img2text_recall_at_k)
        print(text2img_recall_at_k)
        
    print('The mean img2text R@K: {}; The mean text2img R@K: {}'
          .format(img2text_recall_at_k['i2t_r_mean'], text2img_recall_at_k['t2i_r_mean']))
    return {'r@1': (img2text_recall_at_k['i2t_r1'] + text2img_recall_at_k['t2i_r1']) / 2}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme_Text_Retrieval_CLIP training', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)