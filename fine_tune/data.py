import logging
from dataclasses import dataclass
import json
import pandas as pd
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
        return images, texts, str(self.images[idx]), str(self.meme_captions[idx])

class MemeConfigDataset(Dataset):
    def __init__(self, input_filename, transforms, caption_preprocess, text_type='meaning of the meme', tokenizer=None, root=None):
        logging.debug(f'Loading meme configs data from {input_filename}.')
        self.images = []
        self.meme_captions = []
        self.root = root
        if text_type == 'meme_captions':
            text_type='meaning of the meme'
        elif text_type == 'img_captions':
            text_type='visual elaboration'

        assert input_filename.endswith('.json')
        with open(root + input_filename, 'r', encoding='utf-8') as json_file:
            memes_configs = json.load(json_file)
            for conf in tqdm(memes_configs):
                image = root + conf['image_dir'].split("..")[1]

                if image.endswith(('.png', '.jpg', '.jpeg')):
                    if isinstance(conf[text_type], list):
                        if caption_preprocess == 'merge':
                            meme_cap = " ".join(conf[text_type])
                            self.images.append(image)
                            # self.meme_captions.append(meme_cap + ". " +str(conf['meaning of the meme']))
                            self.meme_captions.append(meme_cap)
                        
                        elif caption_preprocess == 'extend':
                            for ite in conf[text_type]:
                                self.images.append(image)
                                self.meme_captions.append(ite)
                                
                        elif caption_preprocess == 'first':
                            self.images.append(image)
                            # self.meme_captions.append(str(conf[text_type][0]) + ". " + str(conf['meaning of the meme']))
                            self.meme_captions.append(str(conf[text_type][0]))
                    else:
                        self.images.append(image)
                        self.meme_captions.append(conf[text_type])

                    
        self.transforms = transforms
        print("Text examples:")
        print(self.meme_captions[0])
        print(self.meme_captions[1])
        logging.debug('Done loading data.')

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.meme_captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenizer(str(self.meme_captions[idx]))
        return images, texts, str(self.images[idx]), str(self.meme_captions[idx])


class FigmemesDataset(Dataset):
    def __init__(self, transforms, 
                text_type='meaning of the meme', 
                tokenizer=None, root=None):
        self.root = root
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.meme_captions, self.images = self.load_figmeme_dataset()
        
    
    def load_figmeme_dataset(self):
        # Simple Way to Read TSV Files in Python using pandas
        standard_spit = pd.read_csv('../data/figmemes/standard_split.tsv', sep='\t')
        # standard_spit = pd.read_csv('../data/figmemes/template_based_memes.tsv', sep='\t')
        # standard_spit = pd.read_csv('../data/figmemes/template_based_instances_test_split.tsv', sep='\t')
        ocrs = pd.read_csv('../data/figmemes/figmemes_ocrs.tsv', sep='\t', index_col=0)
        annotations = pd.read_csv('../data/figmemes/figmemes_annotations.tsv', sep='\t')
        device_columns = ['allusion', 'exaggeration', 'irony', 'anthrop', 'metaphor', 'contrast']
        all_annotations = dict()
        # Iterate over each row in the DataFrame
        for _, row in annotations.iterrows():
            img_annotation = []
            # For each literary device, check if it's labeled as 1
            for device in device_columns:
                if row[device] == 1:
                    # Append a sub-list with the file name (inside a list) and the device name
                    img_annotation.append(device)
            all_annotations[row['img_name']] = img_annotation
            # print(row['img_name'], img_annotation)


        # Select rows where any column contains the keyword "test"
        filtered_df = standard_spit[standard_spit.apply(lambda row: row.astype(str).str.contains('test').any(), axis=1)]
        testset = filtered_df.values.tolist()
        literary_devices = []
        img_dirs = []
        print(standard_spit.columns)
        for ite in testset:
            if len(all_annotations[ite[0]]) > 0 and ite[0] in ocrs.index:
                if 'anthrop' in all_annotations[ite[0]][0]:
                    literary_devices.append(str(ocrs.loc[ite[0], 'text'])) # 'anthropomorphism'+ '. ' + 
                else:
                    literary_devices.append(str(ocrs.loc[ite[0], 'text'])) #  # all_annotations[ite[0]][0]+ '. ' + 
                img_dirs.append(self.root + ite[0])
        
        return literary_devices, img_dirs


    def __len__(self):
        return len(self.meme_captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenizer(str(self.meme_captions[idx]))
        return images, texts, str(self.images[idx]), str(self.meme_captions[idx])

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
