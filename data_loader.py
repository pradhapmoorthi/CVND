import os
import json
import random

import nltk
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from vocabulary import Vocabulary


def get_loader(
    transform,
    mode="train",
    batch_size=1,
    vocab_threshold=None,
    vocab_file="./vocab.pkl",
    start_word="<start>",
    end_word="<end>",
    unk_word="<unk>",
    vocab_from_file=True,
    num_workers=0,
    coco_root="/content/coco",   # ✅ NEW: COCO dataset root
):
    """
    Returns the data loader.

    Args:
        transform: Image transform.
        mode: One of {'train','val','test'}.
        batch_size: Batch size (if in test mode, must have batch_size=1).
        vocab_threshold: Minimum word count threshold.
        vocab_file: File containing the vocabulary.
        start_word/end_word/unk_word: Special tokens.
        vocab_from_file: If False, create vocab from captions file & overwrite any existing vocab_file.
        num_workers: Number of subprocesses to use for data loading.
        coco_root: Root directory of COCO dataset, e.g. '/content/coco'
                  Expected layout:
                    coco_root/train2014
                    coco_root/val2014
                    coco_root/test2014 (optional)
                    coco_root/annotations/*.json
    """
    assert mode in ["train", "val", "test"], "mode must be one of 'train', 'val', or 'test'."
    if vocab_from_file is False:
        assert mode == "train", "To generate vocab, must be in training mode (mode='train')."

    # --- Resolve image folder and annotation file based on mode ---
    ann_dir = os.path.join(coco_root, "annotations")

    if mode == "train":
        if vocab_from_file:
            assert os.path.exists(vocab_file), (
                "vocab_file does not exist. Set vocab_from_file=False to create it."
            )
        img_folder = os.path.join(coco_root, "train2014")
        annotations_file = os.path.join(ann_dir, "captions_train2014.json")

    elif mode == "val":
        # validation uses val2014 captions
        assert os.path.exists(vocab_file), "Must have vocab.pkl (create during training first)."
        img_folder = os.path.join(coco_root, "val2014")
        annotations_file = os.path.join(ann_dir, "captions_val2014.json")

    else:  # mode == "test"
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file is True, "Set vocab_from_file=True for test mode."
        img_folder = os.path.join(coco_root, "test2014")
        annotations_file = os.path.join(ann_dir, "image_info_test2014.json")

    # --- Fail fast if files/folders are missing ---
    assert os.path.exists(img_folder), f"Image folder not found: {img_folder}"
    assert os.path.exists(annotations_file), f"Annotations file not found: {annotations_file}"

    # --- Build dataset ---
    dataset = CoCoDataset(
        transform=transform,
        mode=mode,
        batch_size=batch_size,
        vocab_threshold=vocab_threshold,
        vocab_file=vocab_file,
        start_word=start_word,
        end_word=end_word,
        unk_word=unk_word,
        annotations_file=annotations_file,
        vocab_from_file=vocab_from_file,
        img_folder=img_folder,
    )

    # --- Build loader ---
    if mode == "train":
        # Randomly sample a caption length, then sample indices with that length.
        indices = dataset.get_train_indices()
        sampler = data.sampler.SubsetRandomSampler(indices=indices)
        batch_sampler = data.sampler.BatchSampler(
            sampler=sampler, batch_size=dataset.batch_size, drop_last=False
        )
        data_loader = data.DataLoader(
            dataset=dataset, num_workers=num_workers, batch_sampler=batch_sampler
        )
    else:
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=dataset.batch_size,
            shuffle=(mode != "test"),
            num_workers=num_workers,
        )

    return data_loader


class CoCoDataset(data.Dataset):
    def __init__(
        self,
        transform,
        mode,
        batch_size,
        vocab_threshold,
        vocab_file,
        start_word,
        end_word,
        unk_word,
        annotations_file,
        vocab_from_file,
        img_folder,
    ):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.img_folder = img_folder

        # Vocabulary constructed from captions in training, loaded otherwise
        self.vocab = Vocabulary(
            vocab_threshold, vocab_file, start_word, end_word, unk_word, annotations_file, vocab_from_file
        )

        if self.mode in ["train", "val"]:
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())

            print("Obtaining caption lengths...")
            all_tokens = [
                nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[idx]]["caption"]).lower())
                for idx in tqdm(np.arange(len(self.ids)))
            ]
            self.caption_lengths = [len(tokens) for tokens in all_tokens]

        else:  # test
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item["file_name"] for item in test_info["images"]]

    def __getitem__(self, index):
        if self.mode in ["train", "val"]:
            ann_id = self.ids[index]
            caption_str = self.coco.anns[ann_id]["caption"]
            img_id = self.coco.anns[ann_id]["image_id"]
            file_name = self.coco.loadImgs(img_id)[0]["file_name"]

            image = Image.open(os.path.join(self.img_folder, file_name)).convert("RGB")
            image = self.transform(image)

            tokens = nltk.tokenize.word_tokenize(str(caption_str).lower())
            caption = [self.vocab(self.vocab.start_word)]
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.tensor(caption).long()

            return image, caption

        else:
            file_name = self.paths[index]
            pil_image = Image.open(os.path.join(self.img_folder, file_name)).convert("RGB")
            orig_image = np.array(pil_image)
            image = self.transform(pil_image)
            return orig_image, image

    def get_train_indices(self):
        # Sample a random caption length and pick batch_size examples with that length
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where(np.array(self.caption_lengths) == sel_length)[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode in ["train", "val"]:
            return len(self.ids)
        return len(self.paths)
