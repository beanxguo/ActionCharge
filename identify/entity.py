from torch.utils.data import Dataset as TorchDataset
from typing import List
from collections import OrderedDict
from identify import sample



class EntityType:
    def __init__(self, identifier, index):
        self._identifier = identifier
        self._index = index

    @property
    def identifier(self):
        return self._identifier
    @property
    def index(self):
        return self._index
class AccusationType:
    def __init__(self, identifier, pharse, index):
        self._index = index
        self._identifier = identifier
        self._pharse=pharse
    @property
    def pharse(self):
        return self._pharse
    @property
    def identifier(self):
        return self._identifier
    @property
    def index(self):
        return self._index
class Token:
    def __init__(self, tid: int, index: int, span_start: int, span_end: int, phrase: str):
        self._tid = tid  # ID within the corresponding dataset
        self._index = index
        self._span_start = span_start  # start of token span in document (inclusive)
        self._span_end = span_end
        self._phrase = phrase

    @property
    def index(self):
        return self._index
    @property
    def span_start(self):
        return self._span_start

    @property
    def span_end(self):
        return self._span_end

    @property
    def phrase(self):
        return self._phrase

    def __len__(self):
        return len(self._tokens)
    def __iter__(self):
        return iter(self._tokens)
class Entity:
    def __init__(self, eid: int, entity_type: EntityType, tokens: List[Token], phrase: str):
        self._eid = eid  # ID within the corresponding dataset
        self._entity_type = entity_type
        self._tokens = tokens
        self._phrase = phrase

    def as_tuple_token(self):
        return self._tokens[0].index, self._tokens[-1].index + 1, self._entity_type

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span_token(self):
        return self._tokens[0].index, self._tokens[-1].index + 1

    @property
    def span_context(self):
        return self.span_start, self.span_end

    @property
    def phrase(self):
        return self._phrase

    def __str__(self):
        return self._phrase
class Boundarystart:
    def __init__(self, bsid: int,token_lock: int, boundary_type: int, token: Token, phrase: str):
        self._bsid = bsid
        self._token_lock = token_lock
        self._boundary_type = boundary_type
        self._token = token
        self._phrase = phrase
    @property
    def token_lock(self):
        return self._token_lock

    @property
    def span_start(self):
        return self._token[0].span_start

    @property
    def span_end(self):
        return self._token[-1].span_end

    @property
    def span_token(self):
        return self._token[0].index, self._token[-1].index + 1

    @property
    def span_context(self):
        return self.span_start, self.span_end
class Boundaryend:
    def __init__(self, bsid: int,token_lock: int, boundary_type: int, token: Token, phrase: str):
        self._bsid = bsid
        self._token_lock = token_lock
        self._boundary_type = boundary_type
        self._token = token
        self._phrase = phrase
    @property
    def token_lock(self):
        return self._token_lock

    @property
    def span_start(self):
        return self._token[0].span_start

    @property
    def span_end(self):
        return self._token[-1].span_end

    @property
    def span_token(self):
        return self._token[0].index, self._token[-1].index + 1

    @property
    def span_context(self):
        return self.span_start, self.span_end

class Accusation:
    def __init__(self,aid:int, acc_type: AccusationType, phrase: str):
        self._aid = aid
        self._acc_type=acc_type
        self._pharse = phrase

    def as_acc_tuple(self):
        return self._acc_type.index,self._acc_type
    @property
    def acc_type(self):
        return self._acc_type

class Document:
    def __init__(self,doc_id:int,doc_tokens:List[Token],encoding:List[int],entities: List[Entity],boundarystart:List[Boundarystart], boundaryend:List[Boundaryend],acc:List[Accusation]):
        self._doc_id = doc_id
        self._tokens = doc_tokens
        self._encoding = encoding
        self._entities = entities
        self._boundarystart = boundarystart
        self._boundaryend =boundaryend
        self._accusation=acc

    @property
    def encoding(self):
        return self._encoding

    @property
    def tokens(self):
        return self._tokens

    @property
    def entities(self):
        return self._entities

    @property
    def boundarystart(self):
        return self._boundarystart

    @property
    def boundaryend(self):
        return self._boundaryend

    @property
    def accusation(self):
        return self._accusation



class Dataset(TorchDataset):
    def __init__(self, label, entity_type,accusation_type):
        self._label = label
        self._entity_type = entity_type
        self._accusation_type = accusation_type
        self._documents = OrderedDict()
        self._token=OrderedDict()
        self._entities = OrderedDict()
        self._boundarystart = OrderedDict()
        self._boundaryend = OrderedDict()
        self._accusations = OrderedDict()

        self._doc_id = 0
        self._eid = 0
        self._tid = 0
        self._aid = 0
        self._bsid = 0
        self._beid = 0

    @property
    def label(self):
        return self._label

    def create_token(self,idx,span_start,span_end,phrase,) ->Token:
        token = Token(self._tid, idx, span_start, span_end, phrase)
        self._token[self._tid]=token
        self._tid += 1
        return token
    def create_entity(self, entity_type, tokens, phrase) -> Entity:
        mention = Entity(self._eid, entity_type, tokens, phrase)
        self._entities[self._eid] = mention
        self._eid += 1
        return mention
    def create_boundarystart(self, token_local,boundary_type, token, phrase) -> Boundarystart:
        mention = Boundarystart(self._bsid, token_local, boundary_type, token, phrase)
        self._boundarystart[self._bsid] = mention
        self._bsid += 1
        return mention
    def create_boundaryend(self, token_local,boundary_type, token, phrase) -> Boundaryend:
        mention = Boundaryend(self._beid, token_local, boundary_type, token, phrase)
        self._boundaryend[self._beid] = mention
        self._beid += 1
        return mention
    def create_accusation(self,acc_type,phrase) ->Accusation:
        ac = Accusation(self._aid,acc_type,phrase)
        self._accusations[self._aid] = ac
        return ac
    def create_document(self,doc_tokens,doc_encoding,entities,boundarystart, boundaryend,accusat):
        document = Document(self._doc_id, doc_tokens, doc_encoding,entities,boundarystart, boundaryend,accusat)
        self._documents[self._doc_id] = document
        self._doc_id += 1
        return document

    @property
    def documents(self):
        return list(self._documents.values())
    @property
    def document_count(self):
        return len(self._documents)
    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index: int):
        doc = self._documents[index]
        if self._label=='train':
            return sample.create_train_sample(doc)
        else:
            return sample.create_eval_sample(doc)