import logging
from dataclasses import dataclass
import json

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

class MemecapDataset(Dataset):
    def __init__(self, input_filename, transforms, caption_preprocess, text_type='meme_captions', tokenizer=None, root=None):
        logging.debug(f'Loading meme configs data from {input_filename}.')
        self.images = []
        self.meme_captions = []
        self.root = root
        assert input_filename.endswith('.json')

        memes_img_dir = root + '/data/memecap/memes/memes/'

        with open(root + input_filename, 'r', encoding='utf-8') as json_file:
            memes_configs = json.load(json_file)

        for conf in tqdm(memes_configs):
                image = memes_img_dir + conf['img_fname']
                if image.endswith(('.png', '.jpg', '.jpeg')):
                    if isinstance(conf[text_type], list):
                        if caption_preprocess == 'merge':
                            meme_cap = " ".join(conf[text_type])
                            self.images.append(image)
                            self.meme_captions.append(meme_cap)
                        
                        elif caption_preprocess == 'extend':
                            for ite in conf[text_type]:
                                self.images.append(image)
                                self.meme_captions.append(ite)
                                
                        elif caption_preprocess == 'first':
                            self.images.append(image)
                            self.meme_captions.append(conf[text_type][0])
                    else:
                        self.images.append(image)
                        self.meme_captions.append(conf[text_type])
        
        self.transforms = transforms
        self.tokenizer = tokenizer
        print("Text examples:")
        print(self.meme_captions[0])
        print(self.meme_captions[1])
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.meme_captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenizer(str(self.meme_captions[idx]))
        return images, texts

class MemeConfigDataset(Dataset):
    def __init__(self, input_filename, transforms, tokenizer=None, root=None):
        logging.debug(f'Loading meme configs data from {input_filename}.')
        self.images = []
        self.meme_captions = []
        self.root = root
        assert input_filename.endswith('.json')
        with open(root + input_filename, 'r', encoding='utf-8') as json_file:
            memes_configs = json.load(json_file)
            for conf in tqdm(memes_configs):
                image = root + conf['image_dir'].split("..")[1]
                meme_cap = conf['meaning of the meme']
                if image.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(image)
                    self.meme_captions.append(meme_cap)
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.meme_captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenizer(str(self.meme_captions[idx]))
        return images, texts


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None

    def set_epoch(self, epoch):
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset(args, dataset, is_train):
    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    print('====shuffle====: ', shuffle)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_data(args, preprocess_fns, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    assert args.train_data
    training_dataset = MemeConfigDataset(
            args.train_data,
            preprocess_train,
            root=args.root,
            tokenizer=tokenizer
        )
    validation_dataset = MemeConfigDataset(
            args.val_data,
            preprocess_val,
            root=args.root,
            tokenizer=tokenizer
        )
    test_dataset = MemecapDataset(
        args.test_data,
        preprocess_val,
        caption_preprocess = args.caption_preprocess,
        root=args.root,
        tokenizer=tokenizer
    )
    data = {"train": get_dataset(args, training_dataset, is_train=True),
            "val": get_dataset(args, validation_dataset, is_train=False),
            "test": get_dataset(args, test_dataset, is_train=False),
            }

    return data
