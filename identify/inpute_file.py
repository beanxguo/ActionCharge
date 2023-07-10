from abc import abstractmethod, ABC
from transformers import BertTokenizer
from collections import OrderedDict
from tqdm import tqdm
import json
from identify.entity import EntityType,AccusationType,Dataset,Document



class BaseInputReader(ABC):
    def __init__(self,tokenizer: BertTokenizer,accusation_type:str):
        accu_types = json.load(open(accusation_type), object_pairs_hook=OrderedDict)
        self._entity_type = OrderedDict()
        self._idx2entity_type = OrderedDict()
        self._accusation_type = OrderedDict()
        self._idx2accusation_type = OrderedDict()

        none_entity_type = EntityType('None',0)
        realy_entity_type = EntityType('pred',1)
        self._entity_type['None'] = none_entity_type
        self._entity_type['pred'] = realy_entity_type
        self._idx2entity_type[0] = none_entity_type
        self._idx2entity_type[1] = realy_entity_type


        for i,(key,value)in enumerate(accu_types['accusition'].items()):
            accusation=AccusationType(key,value,i)
            self._accusation_type[key]=accusation
            self._idx2accusation_type[i]=accusation

        self._traindataset=None
        self._evaldataset=None
        self._tokenizer = tokenizer
        self._vocabulary_size = tokenizer.vocab_size
        self._traincontext_size = -1
        self._evalcontext_size = -1


    @property
    def entity_type_count(self):
        return len(self._entity_type)
    @property
    def accusition_type_count(self):
        return len(self._accusation_type)
    def get_traindataset(self) -> Dataset:
        return self._traindatase
    def get_evaldataset(self) ->Dataset:
        return self._evaldataset
    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity
    def get_accusition_type(self, idx) -> AccusationType:
        accusition = self._idx2accusation_type[idx]
        return accusition


class JsonInputReader(BaseInputReader):
    def __init__(self,tokenizer:BertTokenizer,accusation_type:str):
        super().__init__(tokenizer,accusation_type)
    def read(self,dataset_path,dataset_label):
        if dataset_label == "train":
            dataset = Dataset(dataset_label,self._entity_type,self._accusation_type)
            self._parse_dataset(dataset_path, dataset)
            self._traindataset = dataset
            self._traincontext_size = self._calc_context_size(self._traindataset)
        else:
            dataset = Dataset(dataset_label, self._entity_type,self._accusation_type)
            self._parse_dataset(dataset_path, dataset)
            self._evaldataset = dataset
            self._evalcontext_size = self._calc_context_size(self._evaldataset)

    def _parse_dataset(self, dataset_path, dataset):
        documents = open(dataset_path, "r", encoding="utf-8").readlines()

        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):

            self._parse_document(document, dataset)

    def _parse_document(self, doc, dataset) -> Document:

        node_dict = json.loads(doc)

        text = node_dict['token']
        jindex = node_dict['entities']
        accusation = node_dict['accusation']
        text_token=[]
        for word in text:
            text_token.append(word)

        doc_tokens,doc_encoding=self._parse_tokens(text_token, dataset)
        entities = self._parse_entities(jindex, doc_tokens, dataset)
        boundarystart, boundaryend = self._parse_boundary(jindex,doc_tokens,dataset)
        accusat = self._parse_accusation(accusation,dataset)
        document = dataset.create_document(doc_tokens,doc_encoding,entities,boundarystart, boundaryend,accusat)
        return document

    def _parse_tokens(self, jtokens, dataset):
        doc_tokens = []
        doc_encoding = [self._tokenizer.convert_tokens_to_ids('[CLS]')]
        for i, token_phrase in enumerate(jtokens):
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))
            token = dataset.create_token(i, span_start, span_end, token_phrase)
            doc_tokens.append(token)
            doc_encoding += token_encoding
        doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]
        return doc_tokens, doc_encoding

    def _parse_entities(self, jindex, doc_tokens, dataset):
        entities = []

        for j in jindex:
            entity_type = self._entity_type['pred']
            start = j[0]
            end = j[-1]
            tokens = doc_tokens[start:end + 1]
            phrase = "".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase)
            entities.append(entity)
        return entities
    def _parse_boundary(self,jindex,doc_tokens, dataset):
        boundarystarts=[]
        boundaryends=[]
        for j in jindex:
            start = j[0]
            tokens=doc_tokens[start:start+1]
            phrase = "".join([t.phrase for t in tokens])
            boundarystart=dataset.create_boundarystart(start,1,tokens,phrase)
            boundarystarts.append(boundarystart)

            end = j[-1]
            tokens = doc_tokens[end:end+1]
            phrase = "".join([t.phrase for t in tokens])
            boundaryend = dataset.create_boundaryend(end, 1, tokens, phrase)
            boundaryends.append(boundaryend)
        return boundarystarts,boundaryends
    def _parse_accusation(self,accusation,dataset):
        acc=[]
        for a in accusation:
            acc_type = self._accusation_type[a]
            pharse = "".join(a)
            accusat = dataset.create_accusation(acc_type,pharse)
            acc.append(accusat)
        return acc
    def get_dataset(self, label) -> Dataset:
        if label=='train':
            return self._traindataset
        else:
            return self._evaldataset

    def _calc_context_size(self, dataset: Dataset):
        sizes =[]
        for doc in dataset.documents:
            sizes.append(len(doc.encoding))
        context_size =max(sizes)
        return context_size