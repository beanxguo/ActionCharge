from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel
import torch
import random
from torch import nn as nn
from identify import util
import torch.nn.functional as F
from .torch_gcn import GCN
from . build_graph import build_graph

def create_boundary_mask(entities, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    for en in entities:
        mask[en[0]:en[1]] = 1
    return mask
def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask
def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h
def decode_crf_pre_entity(pre_label,token_masks,context_mask):
    device = token_masks.device
    #device=pre_label.device
    bach_size=token_masks.shape[0]
    entity_spans_token=[]
    entity_spans_context=[]
    entity_masks_context=[]
    entity_masks_token=[]
    entity_sample=[]
    for i in range(bach_size):
        context_size = int(context_mask[i].sum().item())
        token_size = int(token_masks[i].sum().item())
        line_pre_label = pre_label[i]
        line_entities_span_token = []
        line_entities_span_context=[]
        line_entity_masks_context = []
        line_entity_masks_token = []
        line_entity_sample = []
        for j in range(1,len(line_pre_label)):
            start = -1
            end = -1
            label=line_pre_label[j]
            if label == 2:
                start = j
                end = j + 1
                for num_back in range(end, len(line_pre_label)-1):
                    if line_pre_label[num_back] == 3:
                        end = num_back + 1
                    else:
                        break
            if start!=-1 and end!=-1:
                line_entities_span_token.append([start-1,end-1])
                line_entities_span_context.append([start,end])
                line_entity_masks_token.append(create_entity_mask(start - 1, end - 1, token_size))
                line_entity_masks_context.append(create_entity_mask(start,end, context_size))
                line_entity_sample.append(1)


        if not line_entities_span_context:
            line_entities_span_token.append([0, 0])
            line_entities_span_token.append([0, 1])
            line_entity_masks_context.append(create_entity_mask(0, 1, context_size))
            line_entity_masks_token.append(create_entity_mask(0, 0, token_size))
            entity_spans_token.append(torch.tensor(line_entities_span_token,dtype=torch.long))
            entity_spans_context.append(torch.tensor(line_entities_span_token,dtype=torch.long))
            entity_masks_context.append(torch.stack(line_entity_masks_context))
            entity_masks_token.append(torch.stack(line_entity_masks_token))
            entity_sample.append(torch.tensor([1], dtype=torch.bool))
        else:
            entity_spans_token.append(torch.tensor(line_entities_span_token,dtype=torch.long))
            entity_spans_context.append(torch.tensor(line_entities_span_context,dtype=torch.long))
            entity_masks_context.append(torch.stack(line_entity_masks_context))
            entity_masks_token.append(torch.stack(line_entity_masks_token))
            entity_sample.append(torch.tensor(line_entity_sample,dtype=torch.bool))

    entity_spans_token=util.padded_stack(entity_spans_token).to(device)
    entity_spans_context=util.padded_stack(entity_spans_context).to(device)
    entity_masks_context=util.padded_stack(entity_masks_context).to(device)
    entity_masks_token=util.padded_stack(entity_masks_token).to(device)
    entity_sample= util.padded_stack(entity_sample).to(device)


    return entity_spans_token,entity_spans_context,entity_masks_context,entity_masks_token,entity_sample




class Identifier(BertPreTrainedModel):
    def __init__(self, config: BertConfig, cls_token: int,entity_types: int,
                 crf_type_count: int, accusation_types: int,
                  prop_drop: float,lstm_drop: float = 0.4, lstm_layers: int = 1, pool_type:str = "max"):
        super(Identifier, self).__init__(config)
        self._cls_token = cls_token
        self._entity_types = entity_types
        self._crf_type_count = crf_type_count
        self._accusation_types = accusation_types
        self.prop_drop=prop_drop
        self.pool_type = pool_type
        config.output_hidden_states = True
        self.bert = BertModel(config)

        lstm_hidden_size = config.hidden_size // 2
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=lstm_hidden_size, num_layers=lstm_layers,
                             bidirectional=True, dropout=lstm_drop,batch_first = True)


        self.dropout = nn.Dropout(self.prop_drop)
        self.entity_classfier =nn.Sequential(
            nn.Linear(3072, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2)
        )
        self.boundarystart_classifier = nn.Sequential(
            nn.Linear(1024, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2)
            # nn.Linear(cls_size, entity_types),
        )
        self.boundaryend_classifier = nn.Sequential(
            nn.Linear(1024, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2)
            # nn.Linear(cls_size, entity_types),
        )
        self.gcn = GCN(
            in_feats=1024,
            n_hidden=200,
            n_classes=self._accusation_types,
            n_layers=1,
            activation=F.elu,
            dropout=0.5
        )
    def _pre_accusition(self,h,encodings,entity_clf,entity_span_context,entity_span_context_mask,entity_span_sample_mask):
        batch_size = entity_span_context_mask.shape[0]
        device = entity_span_context.device
        entity_cls_bool = entity_clf.argmax(dim=-1)*entity_span_sample_mask.long()
        entity_cls_bool=entity_cls_bool.view(entity_cls_bool.shape[0],entity_cls_bool.shape[1],1)
        entity_cls_bool=entity_cls_bool.repeat(1,1,2)
        entities_span_id = entity_cls_bool*entity_span_context
        entities_span_id=entities_span_id.tolist()
        sum_line_entity_mask = []
        for i in range(batch_size):
            line_pre_entities_id = [en for en in entities_span_id[i] if en!=[0,0]]
            mask = torch.zeros(entity_span_context_mask.shape[2], dtype=torch.bool)
            if line_pre_entities_id == None:
                mask[0:1] = 1
            else:
                for span in line_pre_entities_id:
                    mask[span[0]:span[-1]]=1
            sum_line_entity_mask.append(mask)
        sum_line_entity_mask = torch.stack(sum_line_entity_mask)
        sum_line_entity_mask=sum_line_entity_mask.unsqueeze(1).to(device)
        entity_ctx = get_token(h, encodings, self._cls_token)
        accusition_ctx = entity_ctx.unsqueeze(1)
        sum_line_entity_mask=self.combine(h,sum_line_entity_mask,'max')
        embed= [sum_line_entity_mask, accusition_ctx]
        entity_repr_outer = torch.cat(embed, dim=2)
        sum_line_entity_mask=self.dropout(entity_repr_outer)
        pre_accusition = self.accusation_classfier(sum_line_entity_mask)
        return pre_accusition
    def _decode_entity(self,entity_clf,entity_span_token,entity_span_sample_mask):
        batch_size = entity_clf.shape[0]
        device = entity_span_token.device
        entity_cls_bool = entity_clf.argmax(dim=-1) * entity_span_sample_mask.long()
        entity_cls_bool = entity_cls_bool.view(entity_cls_bool.shape[0], entity_cls_bool.shape[1], 1)
        entity_cls_bool = entity_cls_bool.repeat(1, 1, 2)
        entities_span_id = entity_cls_bool * entity_span_token
        entities_span_id = entities_span_id.tolist()
        entity_span = []
        for i in range(batch_size):
            line_pre_entities_id = [en for en in entities_span_id[i] if en!=[0,0]]
            if not line_pre_entities_id:
                line_pre_entities_id.append([0,0])
            entity_span.append(torch.tensor(line_pre_entities_id, dtype=torch.long))
        pre_entity_span_token = util.padded_stack(entity_span).to(device)
        return pre_entity_span_token
    def _boundarystart_class(self,h_context,boundarystart_context_id):
        boundarystart_embeding = util.batch_index(h_context, boundarystart_context_id)
        m = ((boundarystart_context_id == 0).float() * (-1e30)).unsqueeze(-1)
        boundarystart_embeding = m + boundarystart_embeding
        boundarystart_embeding = self.dropout(boundarystart_embeding)
        bs_cls=self.boundarystart_classifier(boundarystart_embeding)
        return bs_cls
    def _boundaryend_class(self,h_context,boundaryend_context_id):
        boundaryend_embeding = util.batch_index(h_context, boundaryend_context_id)
        m = ((boundaryend_context_id == 0).float() * (-1e30)).unsqueeze(-1)
        boundaryend_embeding = m + boundaryend_embeding
        boundaryend_embeding = self.dropout(boundaryend_embeding)
        be_cls=self.boundaryend_classifier(boundaryend_embeding)
        return be_cls
    def _entity_class(self,h_context,entity_span_context,entity_span_context_mask):
        entity_spans_pool = self.combine(h_context, entity_span_context_mask, self.pool_type)
        entity_span_context_inner = entity_span_context.clone()
        entity_span_context_inner[:, :, 0] = entity_span_context_inner[:, :, 0]
        entity_span_context_inner[:, :, 1] = entity_span_context_inner[:, :, 1]-1
        entity_span_context_inner[:, :, 1][entity_span_context_inner[:, :, 1] < 1] = 1
        start_end_embedding_inner = util.batch_index(h_context, entity_span_context_inner)
        start_end_embedding_inner = start_end_embedding_inner.view(start_end_embedding_inner.size(0),
                                                                   start_end_embedding_inner.size(1), -1)
        embed_inner = [start_end_embedding_inner, entity_spans_pool]
        entity_repr_inner = torch.cat(embed_inner, dim=2)
        entity_repr_inner = self.dropout(entity_repr_inner)
        entity_clf = self.entity_classfier(entity_repr_inner)
        return entity_clf

    def _get_entity_span(self,h_context,context_masks,entity_clf,entity_span_context,entity_span_context_mask,entity_span_sample_mask):
        batch_size = entity_span_context_mask.shape[0]
        device = entity_span_context.device
        entity_cls_bool = entity_clf.argmax(dim=-1) * entity_span_sample_mask.long()
        entity_cls_bool = entity_cls_bool.view(entity_cls_bool.shape[0], entity_cls_bool.shape[1], 1)
        entity_cls_bool = entity_cls_bool.repeat(1, 1, 2)
        entities_span_id = entity_cls_bool * entity_span_context
        entities_span_id = entities_span_id.tolist()

        pre_entities_span_context = []
        pre_entities_span_number = []
        pre_entities_sample_mask = []
        pre_entities_span_feature = []
        for i in range(batch_size):
            line_entity_span_context = []
            line_entity_span_context_mask = []
            line_entity_sample = []
            context_size = h_context.shape[1]
            line_pre_entities_id = [en for en in entities_span_id[i] if en!=[0,0]]
            if len(line_pre_entities_id)==0:
                line_entity_span_context.append([0,1])
                line_entity_span_context_mask.append(create_entity_mask(0,1, context_size))
                line_entity_sample.append(1)
            else:
                for span in line_pre_entities_id:
                    line_entity_span_context.append(span)
                    line_entity_span_context_mask.append(create_entity_mask(span[0],span[-1], context_size))
                    line_entity_sample.append(1)

            entity_context_mask=torch.stack(line_entity_span_context_mask)
            entity_context_mask=entity_context_mask.unsqueeze(0).to(device)
            h=h_context[i,:,:].unsqueeze(0)
            entity_spans_pool = self.combine(h, entity_context_mask, self.pool_type)

            pre_entities_span_number.append(len(line_entity_sample))
            pre_entities_sample_mask.append(line_entity_span_context_mask)
            pre_entities_span_context.append(line_entity_span_context)
            pre_entities_span_feature.append(entity_spans_pool.squeeze(0))

        return pre_entities_span_number,pre_entities_sample_mask,pre_entities_span_context,pre_entities_span_feature






    def _creat_span(self,context_masks,bs_cls,be_cls,boundarystart_context_id,boundaryend_context_id,boundarystart_sample_mask,boundaryend_sample_mask,entities_span_context):
        batch_size = bs_cls.shape[0]
        device = context_masks.device
        bs_type_bool=bs_cls.argmax(dim=-1)*boundarystart_sample_mask.long()
        pre_bs_bool=bs_type_bool*boundarystart_context_id
        pre_bs_num=pre_bs_bool.tolist()

        be_type_bool = be_cls.argmax(dim=-1)*boundaryend_sample_mask.long()
        pre_be_bool = be_type_bool*boundaryend_context_id
        pre_be_num=pre_be_bool.tolist()
        entity_span_token = []
        entity_span_context = []
        entity_span_token_mask= []
        entity_span_context_mask = []
        entity_span_type =[]
        entity_span_sample_mask = []
        entities_span_context = entities_span_context.tolist()
        for i in range(batch_size):
            context_size = int(context_masks[i].sum().item())
            token_size=context_size-2
            pre_bs_id=[bs for bs in pre_bs_num[i] if bs!=0]
            pre_be_id=[be for be in pre_be_num[i] if be!=0]

            pre_entity_span_token = []
            pre_entity_span_context = []
            pre_entity_span_token_mask = []
            pre_entity_span_context_mask = []
            pre_entity_span_type = []

            neg_entity_span_token = []
            neg_entity_span_context = []
            neg_entity_span_token_mask = []
            neg_entity_span_context_mask = []
            neg_entity_span_type = []

            gt_entity_span_token = []
            gt_entity_span_context = []
            gt_entity_span_token_mask = []
            gt_entity_span_context_mask = []
            gt_entity_span_type = []

            for span in entities_span_context[i]:
                if span != [0,0]:
                    gt_entity_span_token.append([span[0]-1,span[-1]-1])
                    gt_entity_span_context.append(span)
                    gt_entity_span_token_mask.append(create_entity_mask(span[0]-1,span[-1]-1,token_size))
                    gt_entity_span_context_mask.append(create_entity_mask(*span,context_size))
                    gt_entity_span_type.append(1)

            for bstart in pre_bs_id:
                for span_end in range(1,5):
                    span = [bstart, min(context_size, bstart + span_end)]
                    flag =False
                    for gt_span in gt_entity_span_context:
                        if (span[0]<=gt_span[0] and span[-1]>=gt_span[-1]) or (span[0]>=gt_span[0] and span[-1]<=gt_span[-1]) or (span[0]>=gt_span[0] and span[0]<=gt_span[-1] and span[-1]>=gt_span[0]) or(span[0]<=gt_span[0] and span[-1]>=gt_span[0]  and span[-1]<=gt_span[-1]):
                            flag=True
                            break
                    if flag==True:
                        if (span not in pre_entity_span_context) and (span not in gt_entity_span_context) and (span not in neg_entity_span_context):
                            pre_entity_span_token.append([span[0] - 1, span[-1] - 1])
                            pre_entity_span_context.append(span)
                            pre_entity_span_token_mask.append(create_entity_mask(span[0] - 1, span[-1] - 1, token_size))
                            pre_entity_span_context_mask.append(create_entity_mask(*span, context_size))
                            pre_entity_span_type.append(0)

                    else:
                        if (span not in pre_entity_span_context) and (span not in gt_entity_span_context) and (span not in neg_entity_span_context):
                            neg_entity_span_token.append([span[0] - 1, span[-1] - 1])
                            neg_entity_span_context.append(span)
                            neg_entity_span_token_mask.append(create_entity_mask(span[0] - 1, span[-1] - 1, token_size))
                            neg_entity_span_context_mask.append(create_entity_mask(*span, context_size))
                            neg_entity_span_type.append(0)


            for bend in pre_be_id:
                for span_start in range(0,4):
                    span=[max(1,bend-span_start),min(bend+1,context_size)]
                    flag = False
                    for gt_span in gt_entity_span_context:
                        if (span[0]<=gt_span[0] and span[-1]>=gt_span[-1]) or (span[0]>=gt_span[0] and span[-1]<=gt_span[-1]) or (span[0]>=gt_span[0] and span[0]<=gt_span[-1] and span[-1]>=gt_span[0]) or(span[0]<=gt_span[0] and span[-1]>=gt_span[0]  and span[-1]<=gt_span[-1]):
                            flag = True
                            break
                    if flag == True:
                        if (span not in pre_entity_span_context) and (span not in gt_entity_span_context) and (span not in neg_entity_span_context):
                            pre_entity_span_token.append([span[0] - 1, span[-1] - 1])
                            pre_entity_span_context.append(span)
                            pre_entity_span_token_mask.append(create_entity_mask(span[0] - 1, span[-1] - 1, token_size))
                            pre_entity_span_context_mask.append(create_entity_mask(*span, context_size))
                            pre_entity_span_type.append(0)

                    else:
                        if (span not in pre_entity_span_context) and (span not in gt_entity_span_context) and (span not in neg_entity_span_context):
                            neg_entity_span_token.append([span[0] - 1, span[-1] - 1])
                            neg_entity_span_context.append(span)
                            neg_entity_span_token_mask.append(create_entity_mask(span[0] - 1, span[-1] - 1, token_size))
                            neg_entity_span_context_mask.append(create_entity_mask(*span, context_size))
                            neg_entity_span_type.append(0)

            line_entity_span_token=gt_entity_span_token+pre_entity_span_token
            line_entity_span_context = gt_entity_span_context+pre_entity_span_context
            line_entity_span_token_mask = gt_entity_span_token_mask+pre_entity_span_token_mask
            line_entity_span_context_mask =gt_entity_span_context_mask+pre_entity_span_context_mask
            line_entity_span_type = gt_entity_span_type+pre_entity_span_type
            line_entity_span_sample_mask=torch.ones([len(line_entity_span_token_mask)],dtype=torch.bool)

            entity_span_token.append(torch.tensor(line_entity_span_token,dtype=torch.long))
            entity_span_context.append(torch.tensor(line_entity_span_context,dtype=torch.long))
            entity_span_token_mask.append(torch.stack(line_entity_span_token_mask))
            entity_span_context_mask.append(torch.stack(line_entity_span_context_mask))
            entity_span_type.append(torch.tensor(line_entity_span_type,dtype=torch.long))
            entity_span_sample_mask.append(line_entity_span_sample_mask)

        entity_span_token=util.padded_stack(entity_span_token).to(device)
        entity_span_context = util.padded_stack(entity_span_context).to(device)
        entity_span_token_mask = util.padded_stack(entity_span_token_mask).to(device)
        entity_span_context_mask = util.padded_stack(entity_span_context_mask).to(device)
        entity_span_type=util.padded_stack(entity_span_type).to(device)
        entity_span_sample_mask=util.padded_stack(entity_span_sample_mask).to(device)

        return entity_span_token,entity_span_context,entity_span_token_mask,entity_span_context_mask,entity_span_type,entity_span_sample_mask
    def _creat_eval_span(self,context_masks,bs_cls,be_cls,boundarystart_context_id,boundaryend_context_id,boundarystart_sample_mask,boundaryend_sample_mask):
        batch_size = bs_cls.shape[0]
        device = context_masks.device
        bs_type_bool=bs_cls.argmax(dim=-1)*boundarystart_sample_mask.long()
        pre_bs_bool=bs_type_bool*boundarystart_context_id
        pre_bs_num=pre_bs_bool.tolist()

        be_type_bool = be_cls.argmax(dim=-1)*boundaryend_sample_mask.long()
        pre_be_bool = be_type_bool*boundaryend_context_id
        pre_be_num=pre_be_bool.tolist()
        entity_span_token = []
        entity_span_context = []
        entity_span_token_mask= []
        entity_span_context_mask = []
        entity_span_type =[]
        entity_span_sample_mask = []
        for i in range(batch_size):
            context_size = int(context_masks[i].sum().item())
            token_size=context_size-2
            pre_bs_id=[bs for bs in pre_bs_num[i] if bs!=0]
            pre_be_id=[be for be in pre_be_num[i] if be!=0]

            pre_entity_span_token = []
            pre_entity_span_context = []
            pre_entity_span_token_mask = []
            pre_entity_span_context_mask = []
            pre_entity_span_type = []
            for bstart in pre_bs_id:
                for span_end in range(1,5):
                    span = [bstart, min(context_size, bstart + span_end)]
                    if (span not in pre_entity_span_context) :
                        pre_entity_span_token.append([span[0] - 1, span[-1] - 1])
                        pre_entity_span_context.append(span)
                        pre_entity_span_token_mask.append(create_entity_mask(span[0] - 1, span[-1] - 1, token_size))
                        pre_entity_span_context_mask.append(create_entity_mask(*span, context_size))
                        pre_entity_span_type.append(0)
            for bend in pre_be_id:
                for span_start in range(0,4):
                    span=[max(1,bend-span_start),min(bend+1,context_size)]
                    if (span not in pre_entity_span_context):
                        pre_entity_span_token.append([span[0] - 1, span[-1] - 1])
                        pre_entity_span_context.append(span)
                        pre_entity_span_token_mask.append(create_entity_mask(span[0] - 1, span[-1] - 1, token_size))
                        pre_entity_span_context_mask.append(create_entity_mask(*span, context_size))
                        pre_entity_span_type.append(0)
            line_entity_span_token=pre_entity_span_token
            line_entity_span_context = pre_entity_span_context
            line_entity_span_token_mask = pre_entity_span_token_mask
            line_entity_span_context_mask =pre_entity_span_context_mask
            line_entity_span_type = pre_entity_span_type
            line_entity_span_sample_mask=torch.ones([len(line_entity_span_token_mask)],dtype=torch.bool)

            if not line_entity_span_token:
                line_entity_span_token.append([0,0])
                line_entity_span_context.append([0, 1])
                line_entity_span_context_mask.append(create_entity_mask(0, 1, context_size))
                line_entity_span_token_mask.append(create_entity_mask(0, 0, token_size))
                line_entity_span_type.append(0)

                entity_span_token.append(torch.tensor(line_entity_span_token,dtype=torch.long))
                entity_span_context.append(torch.tensor(line_entity_span_context,dtype=torch.long))
                entity_span_token_mask.append(torch.stack(line_entity_span_token_mask))
                entity_span_context_mask.append(torch.stack(line_entity_span_context_mask))
                entity_span_type.append(torch.tensor(line_entity_span_type,dtype=torch.long))
                entity_span_sample_mask.append(torch.tensor([1], dtype=torch.bool))
            else:
                entity_span_token.append(torch.tensor(line_entity_span_token, dtype=torch.long))
                entity_span_context.append(torch.tensor(line_entity_span_context, dtype=torch.long))
                entity_span_token_mask.append(torch.stack(line_entity_span_token_mask))
                entity_span_context_mask.append(torch.stack(line_entity_span_context_mask))
                entity_span_type.append(torch.tensor(line_entity_span_type, dtype=torch.long))
                entity_span_sample_mask.append(torch.tensor(line_entity_span_sample_mask, dtype=torch.long))
        entity_span_token=util.padded_stack(entity_span_token).to(device)
        entity_span_context = util.padded_stack(entity_span_context).to(device)
        entity_span_token_mask = util.padded_stack(entity_span_token_mask).to(device)
        entity_span_context_mask = util.padded_stack(entity_span_context_mask).to(device)
        entity_span_type=util.padded_stack(entity_span_type).to(device)
        entity_span_sample_mask=util.padded_stack(entity_span_sample_mask).to(device)

        return entity_span_token,entity_span_context,entity_span_token_mask,entity_span_context_mask,entity_span_type,entity_span_sample_mask

    def combine(self, sub, sup_mask, pool_type = "max" ):
        sup = None
        if len(sub.shape) == len(sup_mask.shape) :
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.max(dim=2)[0]
                sup[sup==-1e30]=0
        else:
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub
                sup = sup.max(dim=2)[0]
                sup[sup==-1e30]=0
        return sup
    def _train_forward(self,encodings:torch.tensor,context_masks: torch.tensor,token_masks: torch.tensor,
                       entities_span_context:torch.tensor,
                       boundarystart_context_id:torch.tensor,boundarystart_sample_mask:torch.tensor,
                       boundaryend_context_id:torch.tensor,boundaryend_sample_mask:torch.tensor
                       ):
        h = self.bert(input_ids=encodings, attention_mask=context_masks)[2]
        h = torch.stack(h[-4:], dim=-1).mean(-1)
        batch_size = encodings.shape[0]
        token_count = token_masks.long().sum(-1, keepdim=True)
        context_count =context_masks.long().sum(-1,keepdim=True)
        h_context = nn.utils.rnn.pack_padded_sequence(input=h,lengths=context_count.squeeze(-1).cpu().tolist(),enforce_sorted=False, batch_first=True)
        h_context_o, (_, _) = self.lstm(h_context)
        h_lstm_feature, _ = nn.utils.rnn.pad_packed_sequence(h_context_o,batch_first=True)

        bs_cls =self._boundarystart_class(h_lstm_feature,boundarystart_context_id)
        be_cls = self._boundaryend_class(h_lstm_feature,boundaryend_context_id)
        entity_span_token,entity_span_context,entity_span_token_mask,entity_span_context_mask,entity_span_type,entity_span_sample_mask=self._creat_span(context_masks,bs_cls,be_cls,boundarystart_context_id,boundaryend_context_id,boundarystart_sample_mask,boundaryend_sample_mask,entities_span_context)
        entity_clf=self._entity_class(h_lstm_feature,entity_span_context,entity_span_context_mask)
        device = h.device
        pre_acc_node=[]
        pre_acc_node_ample=[]
        pre_entities_span_number, pre_entities_sample_mask, pre_entities_span_context, pre_entities_span_feature = self._get_entity_span(
            h_lstm_feature, context_masks, entity_clf, entity_span_context, entity_span_context_mask, entity_span_sample_mask)
        for i in range(batch_size):
            g=build_graph(device,pre_entities_span_number[i],pre_entities_span_feature[i])
            gcn_logit = self.gcn(g.ndata['node_feature'],g,g.edata['edge_weight'])
            pre_acc_node_ample.append(torch.ones(gcn_logit.shape[0],dtype=torch.bool))
            gcn_pred=nn.Softmax(dim=1)(gcn_logit)
            pre_acc_node.append(gcn_pred)
        pre_acc_node=util.padded_stack(pre_acc_node).to(device)
        pre_acc_node_ample=util.padded_stack(pre_acc_node_ample).to(device)

        return entity_clf,entity_span_token,entity_span_type,entity_span_sample_mask,bs_cls,be_cls,pre_acc_node,pre_acc_node_ample

    def _eval_forward(self,encodings:torch.tensor,context_masks: torch.tensor,token_masks: torch.tensor,
                       boundarystart_context_id:torch.tensor,boundarystart_sample_mask:torch.tensor,
                       boundaryend_context_id:torch.tensor,boundaryend_sample_mask:torch.tensor):
        h = self.bert(input_ids=encodings, attention_mask=context_masks)[2]
        h = torch.stack(h[-4:], dim=-1).mean(-1)
        batch_size = encodings.shape[0]
        token_count = token_masks.long().sum(-1, keepdim=True)
        context_count = context_masks.long().sum(-1, keepdim=True)

        h_context = nn.utils.rnn.pack_padded_sequence(input=h, lengths=context_count.squeeze(-1).cpu().tolist(),
                                                      enforce_sorted=False, batch_first=True)
        h_context_o, (_, _) = self.lstm(h_context)
        h_lstm_feature, _ = nn.utils.rnn.pad_packed_sequence(h_context_o, batch_first=True)

        bs_cls = self._boundarystart_class(h_lstm_feature, boundarystart_context_id)
        be_cls = self._boundaryend_class(h_lstm_feature, boundaryend_context_id)

        entity_span_token, entity_span_context, entity_span_token_mask, entity_span_context_mask, entity_span_type, entity_span_sample_mask = self._creat_eval_span(context_masks, bs_cls, be_cls, boundarystart_context_id, boundaryend_context_id, boundarystart_sample_mask,boundaryend_sample_mask)
        entity_clf = self._entity_class(h_lstm_feature, entity_span_context, entity_span_context_mask)
        pre_entity_span_token = self._decode_entity(entity_clf,entity_span_token,entity_span_sample_mask)
        pre_entities_span_number, pre_entities_sample_mask, pre_entities_span_context, pre_entities_span_feature = self._get_entity_span(
            h_lstm_feature, context_masks, entity_clf, entity_span_context, entity_span_context_mask, entity_span_sample_mask)
        device = h.device
        pre_acc_node = []
        pre_acc_node_ample = []
        for i in range(batch_size):
            g = build_graph(device, pre_entities_span_number[i], pre_entities_span_feature[i])
            gcn_logit = self.gcn(g.ndata['node_feature'], g, g.edata['edge_weight'])
            pre_acc_node_ample.append(torch.ones(gcn_logit.shape[0], dtype=torch.bool))
            gcn_pred = nn.Softmax(dim=1)(gcn_logit)
            pre_acc_node.append(gcn_pred)
        pre_acc_node = util.padded_stack(pre_acc_node).to(device)
        pre_acc_node_ample = util.padded_stack(pre_acc_node_ample).to(device)

        return pre_acc_node,pre_acc_node_ample,pre_entity_span_token

    def _forward_train(self,encodings:torch.tensor,context_masks: torch.tensor,token_masks: torch.tensor,
                       entities_span_context:torch.tensor,
                       boundarystart_context_id:torch.tensor,boundarystart_sample_mask:torch.tensor,
                       boundaryend_context_id:torch.tensor,boundaryend_sample_mask:torch.tensor):
        return self._train_forward(encodings, context_masks,token_masks,
                                   entities_span_context,
                                   boundarystart_context_id,boundarystart_sample_mask,
                                   boundaryend_context_id,boundaryend_sample_mask
                                   )


    def _forward_eval(self,encodings:torch.tensor,context_masks: torch.tensor,token_masks: torch.tensor,
                       boundarystart_context_id:torch.tensor,boundarystart_sample_mask:torch.tensor,
                       boundaryend_context_id:torch.tensor,boundaryend_sample_mask:torch.tensor):
        return self._eval_forward(encodings, context_masks,token_masks,boundarystart_context_id,boundarystart_sample_mask,boundaryend_context_id,boundaryend_sample_mask)

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)

_MODELS = {
    'identifier': Identifier,
}
def get_model(name):
    return _MODELS[name]