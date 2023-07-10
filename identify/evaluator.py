import torch
from typing import List, Tuple
from sklearn.metrics import precision_recall_fscore_support as prfs
from transformers import BertTokenizer
from identify.entity import Document, Dataset, EntityType,AccusationType
from identify.inpute_file import JsonInputReader

class Evaluator:
    def __init__(self, dataset: Dataset, input_reader: JsonInputReader, text_encoder: BertTokenizer,epoch: int):
        self._dataset = dataset
        self._input_reader = input_reader
        self._text_encoder = text_encoder
        self._epoch = epoch
        self._gt_entities = []
        self._pred_entities = []
        self._gt_accusition=[]
        self._pred_accusition=[]
        self._convert_gt(self._dataset.documents)

    def gt_pre(self):

        return self._gt_accusition, self._pred_accusition

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            gt_entities=doc.entities
            gt_accusation = doc.accusation

            sample_gt_entities = [entity.as_tuple_token() for entity in gt_entities]
            sample_gt_accusition=[acc.as_acc_tuple() for acc in gt_accusation]

            self._gt_entities.append(sample_gt_entities)
            self._gt_accusition.append(sample_gt_accusition)

    def _convert_by_setting(self, gt: List[List[Tuple]], pred: List[List[Tuple]],
                            include_entity_types: bool = True, include_score: bool = False):
        def convert(t):
            if include_entity_types:
                c = list(t[:3])
                return tuple(c)

        converted_gt, converted_pred = [], []
        for sample_gt, sample_pred in zip(gt, pred):
            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])
        return converted_gt, converted_pred

    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False):
        assert len(gt) == len(pred)
        gt_flat = []
        pred_flat = []
        types = set()
        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)
            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(t.index)
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(t.index)
                    types.add(t)
                else:
                    pred_flat.append(0)
        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics

    def _score_acc(self,gt:List[AccusationType],pred:List[AccusationType], print_results: bool = False):
        assert len(gt) == len(pred)
        gt_flat = []
        pred_flat = []
        types=set()
        for sam_acc_gt,sam_acc_pred in zip(gt,pred):
            for acc_g,acc_p in zip(sam_acc_gt,sam_acc_pred):
                g=acc_g[1]
                p=acc_p[1]
                gt_flat.append(g.index)
                pred_flat.append(p.index)
                types.add(g)
                types.add(p)
        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics


    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)
    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        labels = [t.index for t in types]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
        total_support = sum(per_type[-1])
        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

        return [m * 100 for m in micro + macro]
    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        results = [row_fmt % columns, '\n']

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self._get_row(m, t.identifier))
            results.append('\n')

        results.append('\n')

        # micro
        results.append(row_fmt % self._get_row(micro, 'micro'))
        results.append('\n')

        # macro
        results.append(row_fmt % self._get_row(macro, 'macro'))

        results_str = ''.join(results)
        print(results_str)
    def eval_batch(self,pre_accusation: torch.tensor,pre_acc_node_ample:torch.tensor,entity_spans: torch.tensor):
        batch_size = pre_accusation.shape[0]
        line_pre_entity=[]
        for i, line_pre_entity,in enumerate(entity_spans):
            row_pred=[]
            for span in line_pre_entity:
                entity_type=self._input_reader.get_entity_type(1)
                if span[0] == 0 and span[-1] == 0:
                    continue
                row_pred.append(
                    (span[0].tolist(), span[-1].tolist(),entity_type)
                )
            self._pred_entities.append(row_pred)
        pre_acc_type=pre_accusation.argmax(dim=-1)
        pre_acc_type = pre_acc_type.tolist()
        pre_acc_node_ample=pre_acc_node_ample.float().tolist()

        for i in range(batch_size):
            row_acc_pred = []
            label = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0}

            line_pre_sample=pre_acc_node_ample[i]
            line_pre_acc=pre_acc_type[i]
            for l_acc_type,sample in zip(line_pre_acc,line_pre_sample):
                if sample != 0.0:
                    label[l_acc_type] = label[l_acc_type]+1
            acc_type=max(label,key=lambda x: label[x])
            aacu_t=self._input_reader.get_accusition_type(acc_type)
            row_acc_pred.append((acc_type, aacu_t))
            self._pred_accusition.append(row_acc_pred)



    def compute_scores(self):
        print("Evaluation")
        print("")
        print("--- Entities (named entity recognition (NER)) ---")
        print("An entity is considered correct if the entity type and span is predicted correctly")
        gt, pred = self._convert_by_setting(self._gt_entities, self._pred_entities, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=True)

        print("Evaluation")
        print("")
        print("--- Accusition (named entity recognition (NER)) ---")
        print("An Accusition is considered correct if the Accusition type  is predicted correctly")

        self._score_acc(self._gt_accusition, self._pred_accusition, print_results=True)

        return ner_eval