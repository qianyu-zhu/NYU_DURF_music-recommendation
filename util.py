import logging
import random

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn import metrics

import torch
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import os
import pickle


def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()

CACHE_PATH='./cache/'
os.makedirs(CACHE_PATH, exist_ok=True)
def cache_wrapper(name, func, *args):
    path = CACHE_PATH + name
    if os.path.exists(name):
        with open(path, 'rb') as f:
            ret = pickle.load(f) 
            return ret
    ret = func(*args)
    with open(path, 'wb') as f:
        pickle.dump(ret, f)
    return ret

def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True



def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def _print_lr_group(groups, all_keys):
    all_keys.append("Other layers")

    for i, group in enumerate(groups):
        print(f"Keyword:{all_keys[i]} \t lr:{group['lr']} \t \
              wd:{group['weight_decay']} param #:{len(group['params'])}")


        
def build_optimizer(args, model, train_dataset):


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    # [{"key", "learning_rate"}]

    parameter_groups_wd = []
    parameter_groups_wnd = []

    try:
        groups = model.get_grouped_parameters()
    except:
        print("Using old/default grouping")
        groups = [
            {
                'key': 'encoder',
                'lr': args.bert_learning_rate,
            },
            {
                'key': 'classifier',
                'lr': args.learning_rate * 2
            }
        ]

    for group in groups:
        parameter_groups_wd.append({
            "params": [],
            "lr": group['lr'],
            "weight_decay": args.weight_decay,
        })
        parameter_groups_wnd.append({
            "params": [],
            "lr": group['lr'],
            "weight_decay": 0,
        })
    
    # Other groups (i.e. no keywords match)
    parameter_groups_wd.append({
        "params": [],
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
    })

    parameter_groups_wnd.append({
        "params": [],
        "lr": args.learning_rate,
        "weight_decay": 0,
    })
    
    all_keys = [g['key'] for g in groups]

    for n, p in model.named_parameters():
        index = -1
        for i, k in enumerate(all_keys):
            if k in n:
                index = i
                break
        
        if not any(nd in n for nd in no_decay):  # with decay
            parameter_groups_wd[index]['params'].append(p)
        else:
            parameter_groups_wnd[index]['params'].append(p)

    """
    Pretty print lr groups
    """
    print("=" * 60)
    print("With weight dacay")
    _print_lr_group(parameter_groups_wd, all_keys)
    print("With no weight dacay")
    _print_lr_group(parameter_groups_wnd, all_keys)
    print("=" * 60)
    
    print("Using AdamW as optimizer")
    optimizer = torch.optim.AdamW(
        parameter_groups_wd + parameter_groups_wnd, lr=args.learning_rate, eps=args.adam_epsilon
    )
    # optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate)

    # Calculate scheduler number
    data_size = len(train_dataset)

    args.max_steps = data_size * args.config['train_config']['max_epochs'] // args.config['data_config']['dataloader']['batch_size']
    args.warmup_steps = args.max_steps // 10
    print(args.max_steps)
    # from IPython import embed
    # embed()
    # exit(0)
    args.print_steps = args.max_steps / args.max_epochs // 10
    # args.print_steps = 100

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )
    return optimizer, scheduler


def evaluate(predictions, labels, metrics):
    eval_resuts = dict()

    if 'auc' in metrics:
        # prediction and labels are all level-2 class ids
        fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        eval_resuts['auc'] = auc
    return eval_resuts


