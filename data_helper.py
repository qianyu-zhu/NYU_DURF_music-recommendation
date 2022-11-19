import json
import random
from tkinter import FALSE, N
import zipfile
from io import BytesIO
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    RandomSampler,
    SequentialSampler,
    SubsetRandomSampler,
)
from transformers import BertTokenizer, AutoTokenizer, AutoConfig
# from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from PIL import Image
# from category_id_map import category_id_to_lv2id, category_id_to_lv1id
from tqdm import tqdm
import re, math
from torch.utils.data.distributed import DistributedSampler as DS

# from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import scipy
import time, os
import torch.distributed as dist
import torch.utils.data.distributed
from tqdm import tqdm

try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = -1

import pandas as pd
import random
from joblib import Parallel, delayed
import logging
import pickle

from util import cache_wrapper
from utilities.hdf5_getters import open_h5_file_read, get_segments_timbre
import sqlite3
import json

def create_dataloaders(args):

    # if 'TPS' not in args.config['train_csv']:
    # 
    # else:
    if os.path.exists("cache/dataset_dense.pkl"):
        logging.info("Loaded cached dataset")
        with open("cache/dataset_dense.pkl", 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = MusicRatingDataset(args.config['train_csv'], args)
        # dataset = TPS_Dataset(args.config['train_csv'], **args.config['data_config']['dataset'])
        with open("cache/dataset_dense.pkl", 'wb') as f:
            pickle.dump(dataset, f)
        logging.info("Cache dataset")

    all_ids = np.arange(len(dataset))
    np.random.shuffle(all_ids)
    val_index = int((1-args.train_data_ratio) * len(dataset))
    val_ids = all_ids[:val_index]

    # val_ids = np.random.choice(len(dataset), int(0.2 * len(dataset)))
    with open("val_ids.json", 'w') as f:
        f.write(json.dumps(val_ids.tolist()))
    train_ids = list(set(np.arange(len(dataset))) - set(val_ids))
    logging.info("Number of train set %d, number of valid set %d" %(len(train_ids), len(val_ids)))

    train_dataset = Subset(dataset, train_ids)
    val_dataset = Subset(dataset, val_ids)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.config['data_config']['dataloader']['batch_size'],
        shuffle=True,
        pin_memory=args.device == 'cuda'
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.config['data_config']['dataloader']['batch_size'] * 4,
        shuffle=False,
        pin_memory=args.device == 'cuda'
    )


    return train_dataloader, val_dataloader, train_dataset, val_dataset

def user_idx_generator(n_users, batch_users):
    ''' helper function to generate the user index to loop through the dataset
    '''
    for start in range(0, n_users, batch_users):
        end = min(n_users, start + batch_users)
        yield slice(start, end)
def get_n_neg_example(
        neg_candid_map,
        item_ids, 
        uid, neg_num, times_instance_num):

    pos_song_ids = neg_candid_map[uid]
    neg_candidates = item_ids - pos_song_ids
    N = len(pos_song_ids)

    if times_instance_num:
        neg_examples = random.sample(neg_candidates, neg_num * N)
    else:
        neg_examples = random.sample(neg_candidates, neg_num)

    return [uid] * len(neg_examples), neg_examples
def get_n_neg_example_helper(
        neg_candid_map,
        item_ids, 
        user_idx, 
        users_ids, neg_num, times_instance_num):
    uids, negs = [], []
    for uid in users_ids[user_idx]:
        uid, neg = get_n_neg_example(neg_candid_map, item_ids, uid, neg_num, times_instance_num)
        uids += uid
        if not times_instance_num:
            negs.append(neg)
        else:
            negs += neg
    return uids,negs
    # return [uid] * len(neg_examples), neg_candidates

class MusicRatingDataset(Dataset):
    def __init__(self, path, args):
        super().__init__()
        self.df = pd.read_csv(path, header=None)
        self.num_users = len(self.df)
        self.num_items = len(self.df.columns)

        self.binarize = args.config['binarize']
        self.get_records()


    def get_records(self):
        self.records = []
        for uid in tqdm(range(self.num_users)):
            for iid in range(self.num_items):
                rating = self.df.iloc[uid, iid]
                if not np.isnan(rating):
                    if self.binarize:
                        rating = 1 if rating > 5 else 0
                    self.records.append([uid, iid, rating, 0])
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, index):
        return self.records[index]

    def get_num_of_unique(self):
        return [self.num_users, self.num_items]


class TPS_Dataset(Dataset):
    """
    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self, path, binarize=True, neg_sample=2, test_mode=False):
        self.test_mode = test_mode

        self.df = pd.read_csv(path, header=None, sep='\t', names = ['user', 'item', 'count'])

        # Filter untrusted records
        untrusted_song_id = set()
        import re
        pattern = re.compile(r'<(\w+)')
        with open('sid_mismatches.txt', 'r') as f:
            for l in f.readlines():
                untrusted_song_id.add(pattern.findall(l)[0])
        print("Original length of df is %d" % len(self.df))
        self.df = self.df[self.df['item'].isin(untrusted_song_id)]
        print("Filtered length of df is %d" % len(self.df))

        # Numberize string IDs
        self.raw_user_ids = self.df['user'].unique()
        self.user_map = {v:k for k,v in enumerate(sorted(self.raw_user_ids))}
        self.df['user'] = self.df['user'].map(self.user_map)

        self.raw_item_ids = self.df['item'].unique()
        self.item_map = {v:k for k,v in enumerate(sorted(self.raw_item_ids))}
        # self.reverse_item_map = {v:k for k,v in self.item_map.items()}
        self.df['item'] = self.df['item'].map(self.item_map)

        # get track id
        print("Get track ids")
        md_dbfile = 'MSD/AdditionalFiles/track_metadata.db'
        md_dbfile = 'track_metadata.db'
        self.track_ids = []
        with sqlite3.connect(md_dbfile) as conn:
            cur = conn.cursor()
            for sid in tqdm(self.raw_item_ids):
                cur.execute("SELECT track_id FROM songs WHERE song_id = '%s'" % sid)
                tid = cur.fetchone()
                self.track_ids.append(tid)

        # get the normalized ids
        self.user_ids = self.df['user'].unique()
        self.item_ids = set(self.df['item'].unique())

        self.users = self.df['user'].tolist()
        self.items = self.df['item'].tolist()

        if binarize:
            self.labels = [1] * len(self.users)
        else:
            self.labels = self.df['count'].tolist()

        self.neg_candid_map = dict(self.df.groupby('user').apply(lambda x: set(x.item)))

        # Generate neg_sample
        print("Sampling negative examples")
        for user in tqdm(self.user_ids):
            not_candid = self.neg_candid_map[user]
            candid = set(self.item_ids) - set(not_candid)
            negs = np.random.choice(list(candid), neg_sample).tolist()

            self.users += [user] * neg_sample
            self.items += negs
            self.labels += [0] * neg_sample
        
        return


    def get_features(self, number_iid, cut=400):
        tid = self.track_ids[number_iid][0]
        path = "/scratch/work/public/MillionSongDataset/data/%s/%s/%s/%s.h5" % \
            (tid[2], tid[3], tid[4], tid)
        h5 = open_h5_file_read(path)
        timbre = get_segments_timbre(h5)
        if timbre.shape[0] < cut:
            timbre = np.pad(timbre, ((0, cut-timbre.shape[0]), (0, 0)))
        h5.close()
        return timbre[:cut]
        
    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        iid = self.items[idx]
        feature = self.get_features(iid)
        
        return [self.users[idx], iid, self.labels[idx], feature]

        if self.test_mode:
            return {
                'uid': self.users[idx],
                'iid': self.items[idx],
                'neg_iid': self.neg_examples[idx],
                'labels': self.labels[idx],
            }
        else:
            return {
                'uid': self.users[idx],
                'iid': self.items[idx],
                'labels': self.labels[idx],
            }


    def get_num_of_unique(self):
        return [len(self.user_ids), len(self.item_ids)]