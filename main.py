import logging
import os
import time
import torch


os.environ['OPENBLAS_NUM_THREADS'] = '1'

os.makedirs("logs", exist_ok=True)


# from model import MultiModal
from models import *

# from new_model import MultiModal
from config import parse_args
from data_helper import create_dataloaders
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
# from apex import amp
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

def cal_hit(gt_index, pred_indices):
    h = (pred_indices == gt_index).sum(dim=0).float().mean()
    assert h <= 1
    return h


def cal_ndcg(gt_index, pred_items):
    # 
    index = (pred_items==gt_index).nonzero(as_tuple=True)[1]
    
    ndcg = np.reciprocal(np.log2(index+2))

    return ndcg.sum()/pred_items.size()[1]



def validate(args, model, val_dataloader, save_path=None):
    model.eval()
    predictions = []
    labels = []
    losses = []
    uids = []
    iids = []

    hr, ncdg = [], []
    loss_fnt = F.binary_cross_entropy if args.config['binarize'] else F.mse_loss

    with torch.no_grad():
        for batch in tqdm(val_dataloader):

            uid, iid, label, features = [i.to(args.device) for i in batch]

            preds = model(uid, iid, features, logits=not args.config['binarize'])
            label = label.float()

            loss = loss_fnt(preds, label) 
            losses.append(loss.cpu().numpy())

            uids.extend(uid)
            iids.extend(iid)
            predictions.extend(preds.cpu().numpy())
            labels.extend(label.cpu().numpy())
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("uid,iid,label,preds\n")
            for i in range(len(uids)):
                f.write(f"{uids[i]}, {iids[i]}, {labels[i]}, {predictions[i]}\n")



    metrics = []
    if args.config['binarize']:
        metrics.append('auc')
    results = evaluate(predictions, labels, metrics)

    model.train()
    return np.mean(losses), results


def inference(args, model, val_dataloader):
    logging.info("==============================")
    logging.info("Inferencing...")
    logging.info("==============================")

    ckpt_file = args.config['inference']['ckpt_file']
    logging.info("Loading ckpt file " + ckpt_file)
    try:
        model.load_state_dict(torch.load(ckpt_file, map_location=torch.device('cpu'))['model_state_dict'])
    except Exception as e:
        # logging.info("The ckpt loaded doesn't exactly match")
        logging.info(e)
        exit(0)
        logging.info("You may expect this if loaded from pretrained model")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(ckpt_file)['model_state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    loss, results = validate(args,model,val_dataloader, save_path=args.config['inference']['inference_result_save_path'])
    results = {k: round(v, 4) for k, v in results.items()}
    res_str = " ".join([f"{k}: {v}" for k,v in results.items()])
    logging.info(f"Inference results - loss {loss:.3f} " + res_str)



def train_and_validate(args):
    # 1. load data
    #
    #   train_dataset[0] = [uid, sid, ratings, features]
    #
    train_dataloader, val_dataloader, train_dataset, val_dataset = create_dataloaders(args)

    # from IPython import embed
    # embed() or exit(0)

    # 2. build model and optimizers
    user_num, item_num = train_dataset.dataset.get_num_of_unique()
    args.config['model_config'].update( {'user_num': user_num, 'item_num': item_num} )

    logging.info("Num users: %d, num items %d" % (user_num, item_num))
    model_class = eval(args.config['model_type'])
    model = model_class(**args.config['model_config'])

    if 'inference' in args.config and args.config['inference']['active']:
        inference(args, model, val_dataloader)
        return

    if False and os.path.exists(args.ckpt_file):
        pass

    else:
        logging.info("No ckpt file loaded")

    optimizer, scheduler = build_optimizer(args, model, train_dataset)
    if args.device == 'cuda':

        # FP16
        # model, optimizer = amp.initialize(
        #     model.to(args.device), optimizer, enabled=args.fp16, opt_level='O1',
        #     # keep_batchnorm_fp32=True
        # )
        # model = torch.nn.parallel.DataParallel(model)
        model.to(args.device)


    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs

    loss_fnt = F.binary_cross_entropy_with_logits if args.config['binarize'] else F.mse_loss
    for epoch in range(args.config['train_config']['max_epochs']):
        for batch in train_dataloader:
            model.train()

            uid, iid, label, features = [i.to(args.device) for i in batch]

            preds = model(uid, iid, features)
            labels = label.float()
            loss = loss_fnt(preds, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, \
                             learning_rate {scheduler.get_last_lr()[0]}")
            # break

        # 4. validation
        loss, results = validate(args, model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        res_str = " ".join([f"{k}: {v}" for k,v in results.items()])
        logging.info(f"Epoch {epoch} Validation - step {step}: loss {loss:.3f} " + res_str)

        # 5. save checkpoint
        # mean_f1 = results['mean_f1']
        if epoch >= 19: # ncdg > best_score:
            best_score = loss
            state_dict = model.state_dict() if args.device == 'cuda' else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'loss': loss},
                       f'{args.savedmodel_path}/model_{args.config["experiment_name"]}_{args.config["model_type"]}_trainRatio_{args.train_data_ratio}_epoch_{epoch}_loss_{loss}.bin')


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    for i in range(1, 10):
        args.train_data_ratio = 0.1 * i
        logging.info(f"Train dataset ratio: {args.train_data_ratio}")
        train_and_validate(args)


if __name__ == '__main__':
    main()
