import nltk
import os
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from vocabulary import Vocabulary # Assuming Vocabulary class is in vocabulary.py
import json


class CocoDataset(data.Dataset):
    """
    COCO Custom Dataset compatible with torch.utils.data.DataLoader.
    """
    def __init__(self, transform, mode, batch_size, vocab, image_dir='/opt/cocoapi/images/val2014/', annotation_file='/opt/cocoapi/annotations/captions_val2014.json'):
        """
        Args:
            transform: image transformer.
            mode: 'train' or 'test'.
            batch_size: batch size
            vocab: vocabulary wrapper.
            image_dir: path for the directory containing images
            annotation_file: path for json file containing annotations
        """
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = vocab
        self.image_dir = image_dir
        if self.mode == 'train':
            self.coco = COCO(annotation_file)
            self.ids = list(self.coco.anns.keys())
            print('Obtaining caption lengths...')
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in range(len(self.ids))]
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.load(open(os.path.join('/opt/cocoapi/annotations/', 'image_info_test2014.json')))
            self.paths = [item['file_name'] for item in test_info['images']]

    def __getitem__(self, index):
        # For training
        if self.mode == 'train':
            ann_id = self.ids[index]
            caption_text = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']

            image = Image.open(os.path.join(self.image_dir, path)).convert('RGB')
            image = self.transform(image)

            tokens = nltk.tokenize.word_tokenize(str(caption_text).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            # Return image, caption, and caption length
            return image, caption, len(caption) # Added len(caption)

        # For testing
        else:
            path = self.paths[index]

            # Convert image to RGB because some images are grayscale
            PIL_image = Image.open(os.path.join(self.image_dir, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)
            return image, orig_image, path

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        else:
            return len(self.paths)

def get_val_loader(transform, batch_size, vocab, num_workers=1):
    """
    Returns val_loader.
    Args:
        transform: image transformer.
        batch_size: batch size
        vocab: vocabulary wrapper.
        num_workers: number of subprocesses to use for data loading
    """
    # Coco caption validation dataset
    val_coco = CocoDataset(transform=transform,
                          mode='train', # Using 'train' mode of CocoDataset for validation as it provides captions
                          batch_size=batch_size,
                          vocab=vocab,
                          image_dir='/opt/cocoapi/images/val2014/',
                          annotation_file='/opt/cocoapi/annotations/captions_val2014.json')

    # Data loader for COCO validation dataset
    # This will return (images, captions) for each iteration.
    val_loader = torch.utils.data.DataLoader(dataset=val_coco,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=num_workers)
    return val_loader
