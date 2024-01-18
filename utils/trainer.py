# -*- coding: utf-8 -*-
import datetime
import logging
import os

import numpy
import numpy as np
import torch
import torch.nn.utils as nn_utils
from torch._C._autograd import ProfilerActivity
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup


class Trainer(object):
    def __init__(self,
            model, 
            train_loader, 
            dev_loader,
            log_dir, 
            log_steps, 
            validate_steps, 
            num_epochs, 
            lr, 
            warm_up_ratio=0.1, 
            weight_decay=0.01, 
            max_grad_norm=0.5,
        ):

        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.log_dir = log_dir
        self.log_steps = log_steps
        self.validate_steps = validate_steps
        self.num_epochs = num_epochs
        self.lr = lr
        self.warm_up_ratio = warm_up_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        total_steps = len(train_loader) * self.num_epochs
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
            num_warmup_steps=self.warm_up_ratio * total_steps, 
            num_training_steps=total_steps)
        self.best_metric = 0.0
        self.best_type=0.0
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def train(self):
        """
        Train the model.
        """
        logging.info("Total batches per epoch : {}".format(len(self.train_loader)))
        logging.info("Evaluate every {} batches.".format(self.validate_steps))
        best_model_store_path = os.path.join(self.log_dir, "best_model.bin")
        for epoch in range(self.num_epochs):
            logging.info("\nEpoch {}:".format(epoch + 1)) # 下面已经是是把batch打包好的了
            for batch_step, inputs in enumerate(tqdm(self.train_loader)):
                self.model.train()
                with autocast():
                    model_output = self.model(inputs)
                    loss = model_output["loss"]
                loss.backward()
                if self.max_grad_norm > 0:
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if batch_step > 0 and batch_step % self.log_steps == 0:
                    logging.info("Batch Step: {}\tloss: {:.3f}".format(batch_step, loss.item()))

                if batch_step > 0 and batch_step % self.validate_steps == 0:
                    logging.info("Evaluating...")
                    predicts_dict = self.evaluate(loader=self.dev_loader)
                    # logging.info("Evaluation Type Acc: {:.3f} Entity Acc: {:.3f} loss: {:.3f}".format(
                    #     predicts_dict["type_avg_acc"], predicts_dict["entity_avg_acc"], predicts_dict["avg_loss"])
                    # )
                    if predicts_dict["avg_acc"] > self.best_metric:
                        self.best_metric = predicts_dict["avg_acc"]
                        if (predicts_dict['avg_acc'] > self.best_type):
                            self.best_type = predicts_dict['avg_acc']
                        torch.save(self.model, best_model_store_path)
                        logging.info("Saved to [%s]" % best_model_store_path)
            predicts_dict = self.evaluate(
                loader=self.dev_loader)
            if predicts_dict["avg_acc"] > self.best_metric:
                self.best_metric = predicts_dict["avg_acc"]
                if (predicts_dict['avg_acc'] > self.best_type):
                    self.best_type = predicts_dict['avg_acc']
                torch.save(self.model, best_model_store_path)
                logging.info("Saved to [%s]" % best_model_store_path)
            logging.info("Epoch {} training done.".format(epoch + 1))
            model_to_save = os.path.join(self.log_dir, "model_epoch_%d.bin" % (epoch + 1))  # 每个epoch结束保存的模型
            torch.save(self.model, model_to_save)
            logging.info("Saved to [%s]" % model_to_save)

    def evaluate(self, loader):
        self.model.eval()
        type_total_acc = 0.0
        entity_total_acc = 0.0
        loss = []
        for inputs in tqdm(loader):
            with torch.no_grad():
                with autocast():
                    output = self.model(inputs)
                    type_acc = output["type_acc"]
                    entity_acc = output["entity_acc"]
                    type_total_acc += type_acc
                    entity_total_acc += entity_acc
                    loss.append(float(output["loss"]))
        type_avg_acc = np.mean(type_total_acc)
        entity_avg_acc = np.mean(entity_total_acc)
        avg_loss = np.mean(loss)

        avg_acc= type_avg_acc + entity_avg_acc
        return_dict = {
            "type_avg_acc": type_avg_acc,
            "entity_avg_acc": entity_avg_acc,
            "avg_loss": avg_loss,
            "avg_acc": avg_acc
        }
        return return_dict
