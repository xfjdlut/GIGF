# -*- coding: utf-8 -*-
import os
import json
import pickle
from collections import Counter

SEP = "[SEP]"
USER = "[USER]"  # additional special token
BOT = "[BOT]"  # additional special token
NEW_ADD_TOKENS = ["[USER]", "[BOT]"]

def extract_knowledge(kg_list, current_type, current_topic):
    thereshold=0.2
    sub_kg=[]
    if current_topic=='NULL':
        goal = current_type
    else:
        goal=current_type+current_topic
    for triple in kg_list:
        s, p, o = triple
        score_s = calcul_kg(s, goal)
        score_o = calcul_kg(o, goal)
        if (score_o > thereshold or score_s > thereshold):
            sub_kg.append(triple)
    return sub_kg


def calcul_kg(kg,goal):
    """ Calculate char-level f1 score """
    common = Counter(kg) & Counter(goal)
    hit_char_total = sum(common.values())
    kg_char_total = len(kg)
    goal_char_total = len(goal)
    p = hit_char_total / kg_char_total if kg_char_total > 0 else 0
    r = hit_char_total / goal_char_total if goal_char_total > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return f1


def convert_data(fp, extract_kg=False, goal_path=None, type_dic=None, entity_dic=None):
    cur_actions, cur_topics = [], []
    if goal_path is not None:
        with open(goal_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                sample = json.loads(line)
                cur_actions = sample["cur_actions"]
                cur_topics = sample["cur_topics"]
        with open(type_dic, 'r', encoding='utf-8') as ft:
            for line in ft:
                type_dict = json.loads(line)
        with open(entity_dic, 'r', encoding='utf-8') as fe:
            for line in fe:
                entity_dict = json.loads(line)
        type_dict=list(type_dict.keys())
        entity_dict=list(entity_dict.keys())
    data = []
    with open(fp, 'r', encoding='utf-8') as fr:
        for idx, line in enumerate(fr):
            sample = json.loads(line)
            original_goal = sample["original_goal"]
            user_profile = sample["user_profile"]
            history = sample["conversation"]
            resp_str = sample["response"]

            input_str = ""
            for k, v in user_profile.items():
                input_str += k
                input_str += v
                input_str += SEP

            if extract_kg:
                if goal_path is not None:
                    # extract knowledge according to generated plans
                    action = type_dict[cur_actions[idx]]
                    topic = entity_dict[cur_topics[idx]]
                    kg_list = extract_knowledge(sample["knowledge"], action, topic)
                    # kg_list = sample["knowledge"]
                    for triple in kg_list:
                        kd = "".join(triple)
                        input_str += kd
                        input_str += SEP

                    input_str += action + topic + SEP
                else:
                    # extract knowledge according to current labeled topic
                    kg_list = extract_knowledge(sample["knowledge"], sample["action_path"][0], sample["topic_path"][0])
                    for triple in kg_list:
                        kd = "".join(triple)
                        input_str += kd
                        input_str += SEP
                    input_str += sample["action_path"][0] + sample["topic_path"][0] + SEP
            else:
                kg_list = sample["knowledge"]
                for triple in kg_list:
                    kd = "".join(triple)
                    input_str += kd
                    input_str += SEP
                input_str += sample["action_path"][0] + sample["topic_path"][0] + SEP

            led_by_bot = False
            if "Bot主动" in original_goal[0]:
                led_by_bot = True
            for hdx, utt_str in enumerate(history):
                if hdx % 2 == 0:
                    if led_by_bot:
                        input_str += BOT
                    else:
                        input_str += USER
                else:
                    if led_by_bot:
                        input_str += USER
                    else:
                        input_str += BOT
                input_str += utt_str
            input_str += BOT
            input_str+=SEP+'生成回复：'+SEP
            data.append([input_str, resp_str])
    return data


def tokenize(tokenizer, obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(tokenizer, o)) for n, o in obj.items())
    return list(tokenize(tokenizer, o) for o in obj)


def load_data(tokenizer, logger, dataset_path, cache_dir, data_partition="train", use_goal=False, goal_path=None,
              type_dic=None, entity_dic=None):
    """ Load data from cache or create from raw data."""
    if use_goal:
        cache_dir = cache_dir + '_w_Goal'

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_path = os.path.join(cache_dir, "{}_cache.pkl".format(data_partition))

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            logger.info("Loading cached data from [{}]".format(cache_path))
            tokenized_data = pickle.load(f)
    else:
        logger.info("Creating cached data [{}]".format(cache_path))

        if use_goal:
            if data_partition == "test":
                assert goal_path is not None
                logger.info("Loading from [{}] to prepare test data for goal-enhanced generation.".format(goal_path))

                data = convert_data(fp=dataset_path, extract_kg=True, goal_path=goal_path, type_dic=type_dic,
                                    entity_dic=entity_dic)
            else:
                logger.info("Prepare train/valid data for Goal-enhanced generation.")

                data = convert_data(fp=dataset_path, extract_kg=True)
        else:
            # prepare data for GPT2 fine-tuning
            data = convert_data(fp=dataset_path)

            # tokenize data
        logger.info("Tokenizing ...")
        tokenized_data = tokenize(tokenizer, data)

        # caching data
        with open(cache_path, 'wb') as f:
            pickle.dump(tokenized_data, f)


    logger.info("Total of {} instances were cached.".format(len(tokenized_data)))
    return tokenized_data
