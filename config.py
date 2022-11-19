import argparse
import yaml
from yaml.loader import SafeLoader
import logging
import json
import time

folder_title = "baseline"
epoch = 6
# batch_size =  #{frames:bc, 32:28，8:28/40/64-1.28e-5，16:34}
# num_step_per_epoch = 1000000 // batch_size
# warmup_steps = num_step_per_epoch // 5
lr = 0.45e-5 #4e-5 / 100 * batch_size

def process_args(args):
    # data_size = 75000 if "unlabel" not in args.train_annotation else 1000000

    # args.num_step_per_epoch = data_size // args.batch_size

    # args.max_steps = args.num_step_per_epoch * args.max_epochs
    # args.warmup_steps = args.max_steps // 10

    # args.bert_max_steps = args.max_steps
    # args.bert_warmup_steps = args.warmup_steps

    # args.print_steps = args.num_step_per_epoch//30

    # args.val_batch_size = args.batch_size
    # args.test_batch_size = args.batch_size
    if args.config_file:
        with open(args.config_file, "r") as f:
            config = yaml.load(f, SafeLoader)
        # from IPython import embed
        # embed() or exit
        args.config = config
    
        args.savedmodel_path = "save/" + args.config['experiment_name']

    if args.config['inference'] and args.config['inference']['active']:
        inference_flag = 'inference'
    else:
        inference_flag = 'train'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(f"logs/train_{args.config['experiment_name']}_{inference_flag}.log"),
            logging.StreamHandler()
        ]

    )
    logging.info("Configuration File:")
    logging.info(json.dumps(args.config))

    return args

"""
seq_len 256      512    256+clip2
bc:     64       32        40
lr:    1.28e-5  0.85e-5   0.6e-5
"""
def parse_args(parse_text = None):
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=57706989, help="random seed.")
    parser.add_argument("--fold", type=int, default=0, help="n/5 fold")
    # parser.add_argument("--dropout", type=float, default=0.3, help="dropout ratio")

    # ========================= Data Configs ==========================
    parser.add_argument('--train_csv', type=str, default='musicRatings.csv')
    # parser.add_argument('--test_csv', type=str, default='../content_wmf/code/in.test.num.csv')
    # parser.add_argument('--train_csv', type=str, default='ncf/train_more.csv')
    # parser.add_argument('--test_csv', type=str, default='ncf/test_more.csv')
    parser.add_argument('--config_file', type=str, default='configs/ncf_binarize.yml')


    # ========================= Model Configs ==========================


    parser.add_argument('--train_zip_frames', type=str, default='/home/tione/notebook/data/zip_frames/labeled/')
    parser.add_argument('--test_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/test/')
    parser.add_argument('--test_output_csv', type=str, default='/opt/ml/output/result.csv')
    parser.add_argument('--unlabel_zip_frames', type=str, default='/home/tione/notebook/data/zip_frames/unlabeled/')
    parser.add_argument('--unlabel_annotation', type=str, default='/home/tione/notebook/data/annotations/unlabeled_new.json')
    # parser.add_argument('--unlabel_zip_feats', type=str, default='../data/zip_feats/labeled.zip') # debug
    # parser.add_argument('--unlabel_annotation', type=str, default='../data/annotations/labeled.json') # debug
    
    parser.add_argument("--prefetch", default=32, type=int, help="use for training duration per worker")
    parser.add_argument("--num_workers", default=8, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument("--savedmodel_path", type=str, default=f"save/{folder_title}")
    parser.add_argument("--ckpt_file", type=str, default=f"save/{folder_title}/model_.bin")
    parser.add_argument("--swin_ckpt", type=str, default=f"")
    parser.add_argument("--best_score",default=0.66,type=float,help="save checkpoint if mean_f1 > best_score",)

    # ========================= Learning Configs ========================== large
    parser.add_argument("--max_epochs", type=int, default=20, help="How many epochs")
    #parser.add_argument("--max_steps",default=total_steps,type=int,metavar="N",help="number of total epochs to run",)
    # parser.add_argument("--print_steps",type=int,default=num_step_per_epoch//30,help="Number of steps to log training metrics.",)
    # parser.add_argument("--warmup_steps",default=warmup_steps,type=int,help="warm ups for parameters not in bert or vit",)
    parser.add_argument("--minimum_lr", default=1e-10, type=float, help="minimum learning rate")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="initial learning rate")
    parser.add_argument("--bert_learning_rate", default=1e-5, type=float, help="bert learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-12, type=float, help="Epsilon for Adam optimizer.")


    args = parser.parse_args() if parse_text is None else parser.parse_args(parse_text)

    return process_args(args)