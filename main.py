# -*- coding: utf-8 -*
import argparse
import enum
import json
import os
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import logging
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from model.GIGL import GIGL
from utils.dataset import DuRecDialDataset
from utils.data_utils import get_tokenizer, combine_tokens
from utils.data_collator import data_Collator
from utils.trainer import Trainer

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
  handlers=[
      logging.StreamHandler(sys.stdout)
  ]
)
def seed_torch(seed=42): #3
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "2"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    # torch.use_deterministic_algorithms(True)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=["train", "test"])
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    # ==================== Data ====================
    parser.add_argument('--train_data', type=str, default="data/DuRecDial/new_sample_train.json", help="Path of training data")
    parser.add_argument('--dev_data', type=str, default="data/DuRecDial/new_sample_dev.json", help="Path of dev data")
    parser.add_argument('--test_data', type=str, default="data/DuRecDial/new_sample_test.json", help="Path of test data")
    parser.add_argument('--type_dic', type=str, default="data/DuRecDial/type_dict.json", help="Path of type dictionary")
    parser.add_argument('--entity_dic', type=str, default="data/DuRecDial/entity_dict.json", help="Path of entity dictionary")
    parser.add_argument('--bert_dir', type=str, default="config/bert-base-chinese", help="Path of BERT Model")
    parser.add_argument('--cache_dir', type=str, default="caches/goalPlanning", help="Path of the preocessed dara for training, valid and test")
    parser.add_argument('--log_dir', type=str, default="logs/GoalPlanning", help="Path for the trained model")
    parser.add_argument('--use_knowledge_hop', type=str2bool, default="true", help="process the knowledge triplets of input")
    parser.add_argument('--turn_type_size', type=int, default=16, help="process the knowledge triplets of input")

    #==================== Graph =======================
    parser.add_argument('--kc_balance', type=float, default=0.42,
                        help="The alpha value in the paper which is the balance of knowledge graph and co-occurrence graph")
    parser.add_argument('--num_of_layers', type=int, default=2, help="The layers of the Goal Interaction Graph")
    parser.add_argument('--num_heads_per_layer', type=int, default=[8,8,1], help="The heads per layer of the Goal Interaction Graph")
    parser.add_argument('--num_features_per_layers', type=int, default=[768,64,768], help="The features per layers of the Goal Interaction Graph")
    parser.add_argument('--dim_of_edge', type=int, default=64, help="The dimension of the edges in the Goal Interaction Graph")
    parser.add_argument('--feat_drop', type=float, default=0.4,help="dropout of the feature in the Goal Interaction Graph")
    parser.add_argument('--attn_drop', type=float, default=0.5, help="dropout of the attention in the Goal Interaction Graph")
    parser.add_argument('--log_attention_weights', type=str, default=False)

    # ==================== Train ====================
    parser.add_argument('--load_checkpoint', type=str, default=None, help="The path of the checkpoint")
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--log_steps', type=int, default=400)
    parser.add_argument('--validate_steps', type=int, default=2000)
    parser.add_argument('--use_gpu', type=str2bool, default="True")
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warm_up_ratio', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=0.4)
    parser.add_argument('--embed_dim', type=int, default=768, help="")
    parser.add_argument('--ff_embed_dim', type=int, default=3072 , help="The dimension of the FFN layers")
    parser.add_argument('--layers', type=int, default=12, help="The numbers of the decoder layer")
    parser.add_argument('--decoder_layerdrop', type=float, default=0.1 ,help="The dropout rate of the decoder")
    parser.add_argument('--max_position_embeddings', type=int, default=512)
    parser.add_argument('--share_decoder_embedding', type=str2bool, default="False")
    parser.add_argument('--scale_embedding', type=str2bool, default="True")
    parser.add_argument('--init_std', type=float, default=0.02 , help="The initialization parameters of the model")
    parser.add_argument('--decoder_attention_heads', type=int, default=8)
    parser.add_argument('--decoder_layers', type=int, default=12)
    parser.add_argument('--output_attentions', type=bool, default=False)
    parser.add_argument('--output_hidden_states', type=bool, default=False)
    parser.add_argument('--use_cache', type=bool, default=False)
    parser.add_argument('--activation_function', type=str, default="gelu")
    parser.add_argument('--decoder_ffn_dim', type=int, default=3072, help="The dimension of the FFN layers")
    parser.add_argument('--dropout', type=float, default=0.1 , help="The dropout of the decoder and encoder module")

    #==================== Generate ====================
    parser.add_argument('--infer_checkpoint', type=str, default="best_model.bin",help="The path of the checkpoint when testing, the default is log_dir/best_model.bin")
    parser.add_argument('--test_batch_size', type=int, default=6)

   #========================= Record ============================
    parser.add_argument('--test_output', type=str, default="outputs/pred_goal.json",help="The path of the generated goals")

    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif v.lower() in ('false',' no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def print_args(args):
    print("=============== Args ===============")
    for k in vars(args):
        print("%s: %s" % (k, vars(args)[k]))

def set_seed(args):
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

def run_train(args):
    logging.info("=============== Training ===============")
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer, num_added_tokens, token_id_dict = get_tokenizer(config_dir=args.bert_dir)
    args.tokenizer=tokenizer
    args.num_added_tokens=num_added_tokens
    args.token_id_dict=token_id_dict
    args.vocab_size = len(tokenizer)
    args.pad_token_id = token_id_dict["pad_token_id"]
    args.bos_token_id = token_id_dict["bos_token_id"]
    args.eos_token_id = token_id_dict["eos_token_id"]
    if torch.cuda.is_available():
        args.device=torch.device('cuda')
    logging.info("{}: Add {} additional special tokens.".format(type(tokenizer).__name__, num_added_tokens))


    train_dataset = DuRecDialDataset(data_path=args.train_data, type_path=args.type_dic,entity_path=args.entity_dic,tokenizer=tokenizer, data_partition='train',\
        cache_dir=args.cache_dir, turn_type_size=args.turn_type_size)
    dev_dataset = DuRecDialDataset(data_path=args.dev_data,type_path=args.type_dic,entity_path=args.entity_dic, tokenizer=tokenizer, data_partition='dev',\
        cache_dir=args.cache_dir, turn_type_size=args.turn_type_size)

    collator = data_Collator(device=device, padding_idx=args.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator.custom_collate)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator.custom_collate)

    if args.load_checkpoint is not None:
        model = torch.load(args.load_checkpoint)
    else:
        model = GIGL(args=args)
    model.to(device)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total parameters: {}\tTrainable parameters: {}".format(total_num, trainable_num))

    trainer = Trainer(model=model, train_loader=train_loader, dev_loader=dev_loader,
        log_dir=args.log_dir, log_steps=args.log_steps, validate_steps=args.validate_steps, 
        num_epochs=args.num_epochs, lr=args.lr, warm_up_ratio=args.warm_up_ratio,
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm
    )    #初始化Trainer
    trainer.train()


def run_test(args):
    logging.info("=============== Testing ===============")
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer, _, token_id_dict = get_tokenizer(config_dir=args.bert_dir)
    args.pad_token_id = token_id_dict["pad_token_id"]

    test_dataset = DuRecDialDataset(data_path=args.test_data, type_path=args.type_dic,entity_path=args.entity_dic,tokenizer=tokenizer, data_partition="test",
        cache_dir=args.cache_dir,is_test=True,turn_type_size=args.turn_type_size)
    collator = data_Collator(device=device, padding_idx=args.pad_token_id)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=collator.custom_collate)

    # 加载模型
    if args.infer_checkpoint is not None:
        model_path = os.path.join(args.log_dir, args.infer_checkpoint)
    else:
        model_path = os.path.join(args.log_dir, "best_model.bin")
    model = torch.load(model_path)
    logging.info("Model loaded from [{}]".format(model_path))
    model.to(device)
    model.eval()

    type_label=None
    entity_label=None
    type_pred=None
    entity_pred=None

    for idx, inputs in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            with autocast():
                output = model.generate(inputs)
            if (type_label == None):
                type_label = output[0]
            else:
                type_label = torch.cat([type_label, output[0]], dim=0)
            if (entity_label == None):
                entity_label = output[1]
            else:
                entity_label = torch.cat([entity_label, output[1]], dim=0)
            if (type_pred == None):
                type_pred = output[2]
            else:
                type_pred = torch.cat([type_pred, output[2]], dim=0)
            if (entity_pred == None):
                entity_pred = output[3]
            else:
                entity_pred = torch.cat([entity_pred, output[3]], dim=0)
    type_label=type_label.cpu().numpy().tolist()
    type_pred=type_pred.cpu().numpy().tolist()
    entity_label=entity_label.cpu().numpy().tolist()
    entity_pred=entity_pred.cpu().numpy().tolist()

    test_output={}
    test_output['cur_actions']=type_pred
    test_output['cur_topics']=entity_pred
    with open(args.test_output,'w',encoding='utf-8') as f:
        line=json.dumps(test_output,ensure_ascii=False)
        f.write(line + "\n")

    type_accuracy_score=accuracy_score(type_label, type_pred)
    type_recall_score=recall_score(type_label, type_pred, average='macro')
    type_precision_score=precision_score(type_label, type_pred,average='macro')
    type_f1_score=f1_score(type_label, type_pred, average='macro')

    entity_accuracy_score = accuracy_score(entity_label, entity_pred)
    entity_recall_score = recall_score(entity_label,entity_pred, average='macro')
    entity_precision_score = precision_score(entity_label, entity_pred, average='macro')
    entity_f1_score = f1_score(entity_label, entity_pred, average='macro')

    print(type_accuracy_score,type_recall_score,type_precision_score,type_f1_score,entity_accuracy_score,entity_recall_score,entity_precision_score,entity_f1_score)
    return type_accuracy_score,type_recall_score,type_precision_score,type_f1_score,entity_accuracy_score,entity_recall_score,entity_precision_score,entity_f1_score

if __name__ == "__main__":
    args = parse_config()
    set_seed(args)
    seed_torch(42)
    print(torch.cuda.is_available())
    if args.mode == "train":
        print_args(args)
        run_train(args)
    elif args.mode == "test":
        run_test(args)
    else:
        exit("Please specify the \"mode\" parameter!")
