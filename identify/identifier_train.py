import argparse
import transformers
from transformers import BertTokenizer, BertConfig, AdamW
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import os
import datetime
from identify import util
from identify.inpute_file import JsonInputReader,BaseInputReader
from identify.entity import Dataset
from identify import models
from identify.loss import IdentifierLoss
from identify import sample
from identify.evaluator import Evaluator
from identify.focalloss import FocalLoss


class IdentifierTrainer():
    def __init__(self,args:argparse.Namespace):
        self.args=args
        if args.cpu:
            device = "cpu"
        else:
            device = "cuda:" + args.device_id
        if args.seed is not None:
            util.set_seed(args.seed)
        name = str(datetime.datetime.now()).replace(' ', '_')
        self._save_path = os.path.join(self.args.save_path, name)
        self._device = torch.device(device)
        self._tokenizer = BertTokenizer.from_pretrained(self.args.model_path)

    def train(self,train_path: str, valid_path: str, input_reader_cls: BaseInputReader):
        torch.set_num_threads(1)
        input_reader=input_reader_cls(self._tokenizer,self.args.accusation_type)
        input_reader.read(train_path,'train')
        train_dataset = input_reader.get_dataset('train')
        train_sample_count = train_dataset.document_count
        epoch_batch=train_sample_count//self.args.train_batch_size
        total_batch=epoch_batch*self.args.epochs

        model_class = models.get_model(self.args.model_type)
        config = BertConfig.from_pretrained(self.args.model_path)
        model = model_class.from_pretrained(self.args.model_path,config=config,
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            entity_types=input_reader.entity_type_count, crf_type_count=self.args.crf_type_count,accusation_types=input_reader.accusition_type_count,
                                            prop_drop=self.args.prop_drop, lstm_drop=self.args.lstm_drop,
                                            lstm_layers=self.args.lstm_layers, pool_type=self.args.pool_type
                                                )
        model.to(self._device)
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=self.args.lr, weight_decay=self.args.weight_decay, correct_bias=False)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                     num_warmup_steps=self.args.lr_warmup * total_batch,
                                                                     num_training_steps=total_batch)
        self.scheduler = scheduler
        boundarystart_loss = torch.nn.CrossEntropyLoss(reduction='none')
        boundaryend_loss = torch.nn.CrossEntropyLoss(reduction='none')
        entity_loss = FocalLoss(class_num=2, reduction='none',
                                     gamma=self.args.accusation_FL_gamma)
        accusation_criterion = FocalLoss(class_num=input_reader.accusition_type_count, reduction='none',
                                     gamma=self.args.accusation_FL_gamma)
        compute_loss = IdentifierLoss(entity_loss,boundarystart_loss,boundaryend_loss,accusation_criterion, model,optimizer, scheduler,self.args.max_grad_norm)

        best_f1 = -1

        for epoch in range(self.args.epochs):
            self._train_epoch(model,compute_loss, train_dataset,epoch)

            input_reader.read(valid_path, 'valid')
            validation_dataset=input_reader.get_dataset('valid')
            f1=self._eval_epoch(model, validation_dataset, input_reader, epoch)

            if best_f1 < f1[2]:
                print(f"Best F1 score update, from {best_f1} to {f1[2]}")
                best_f1 = f1[2]
                util.save_model(self._save_path, model, self._tokenizer, epoch,
                                 name='best_model')
        util.save_model(self._save_path, model, self._tokenizer, epoch,
                         name='final_model')


    def _train_epoch(self, model: torch.nn.Module, compute_loss: IdentifierLoss, dataset: Dataset,epoch: int):
        data_loader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=False, drop_last=True,
                                  collate_fn=sample.collate_fn_padding)
        model.zero_grad()
        iteration = 0
        total = dataset.document_count // self.args.train_batch_size
        global_loss = 0
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = util.to_device(batch, self._device)
            entity_clf,entity_span_token,entity_span_type,entity_span_sample_mask,bs_cls,be_cls,pre_acc_node,pre_acc_node_ample=model(encodings=batch['encodings'],context_masks=batch['context_masks'],token_masks=batch['token_masks'],
                    entities_span_context=batch['entities_span_context'],
                    boundarystart_context_id=batch['boundarystart_context_id'],boundarystart_sample_mask=batch['boundarystart_sample_mask'],
                    boundaryend_context_id=batch['boundaryend_context_id'],boundaryend_sample_mask=batch['boundaryend_sample_mask']
                  )
            train_loss=compute_loss.compute(entity_clf=entity_clf,entity_span_type=entity_span_type,entity_span_sample_mask=entity_span_sample_mask,
                                            bs_cls=bs_cls,boundarystart_types=batch['boundarystart_types'],boundarystart_sample_mask=batch['boundarystart_sample_mask'],
                                            be_cls=be_cls,boundaryend_types=batch['boundaryend_types'],boundaryend_sample_mask=batch['boundaryend_sample_mask'],
                                            pre_acc_node=pre_acc_node,pre_acc_node_ample=pre_acc_node_ample,gt_accusation=batch['accusation_type'])

    def eval(self, test_path: str, input_reader_cls: BaseInputReader):
        input_reader = input_reader_cls(self._tokenizer, self.args.accusation_type)
        input_reader.read(test_path,'valid')
        model_class = models.get_model(self.args.model_type)
        config = BertConfig.from_pretrained(self.args.model_path)
        model = model_class.from_pretrained(self.args.model_path,config=config,
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            entity_types=input_reader.entity_type_count,
                                            crf_type_count=self.args.crf_type_count,
                                            accusation_types=input_reader.accusition_type_count,
                                            prop_drop=self.args.prop_drop, lstm_drop=self.args.lstm_drop,
                                            lstm_layers=self.args.lstm_layers, pool_type=self.args.pool_type
                                            )
        model.to(self._device)
        validation_dataset = input_reader.get_dataset('valid')
        self._eval_epoch(model, validation_dataset, input_reader)
    def _eval_epoch(self,model: torch.nn.Module, dataset: Dataset, input_reader: JsonInputReader,epoch: int=0):
        evaluator=Evaluator(dataset,input_reader,self._tokenizer,epoch)
        data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False, drop_last=False, collate_fn=sample.collate_fn_padding)
        with torch.no_grad():
            model.eval()
            total = math.ceil(dataset.document_count / self.args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                batch = util.to_device(batch, self._device)
                result=model(encodings=batch['encodings'],context_masks=batch['context_masks'],token_masks=batch['token_masks'],
                    boundarystart_context_id=batch['boundarystart_context_id'],boundarystart_sample_mask=batch['boundarystart_sample_mask'],
                    boundaryend_context_id=batch['boundaryend_context_id'],boundaryend_sample_mask=batch['boundaryend_sample_mask'],
                    evaluate=True
                  )
                pre_acc_node,pre_acc_node_ample,pre_entity_span_token=result
                evaluator.eval_batch(pre_accusation=pre_acc_node,pre_acc_node_ample=pre_acc_node_ample,entity_spans=pre_entity_span_token)


        ner_eval = evaluator.compute_scores()
        return ner_eval
    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            # regressier
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}, ]

        return optimizer_params