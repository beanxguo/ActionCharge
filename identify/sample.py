import torch
from identify import util

def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask
def create_train_sample(doc):
    encodings =doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    pos_boundarstart = doc.boundarystart
    pos_boundarend = doc.boundaryend
    pos_boundarystart_token_id, pos_boundarystart_context_id, pos_boundarystart_types, pos_entity_boundarystart_masks,pos_boundarystart_token_masks,pos_boundarystart_context_masks= [], [], [], [], [], []
    for boundary in pos_boundarstart:
        pos_boundarystart_token_id.append(boundary.token_lock)
        pos_boundarystart_context_id.append(boundary.token_lock+1)
        pos_boundarystart_types.append(1)
        pos_boundarystart_token_masks.append(create_entity_mask(*boundary.span_token, token_count))
        pos_boundarystart_context_masks.append(create_entity_mask(*boundary.span_context,context_size))
    pos_boundaryend_token_id, pos_boundaryend_context_id, pos_boundaryend_types, pos_entity_boundaryend_masks, pos_boundaryend_token_masks, pos_boundaryend_context_masks = [], [], [], [], [], []
    for boundary in pos_boundarend:
        pos_boundaryend_token_id.append(boundary.token_lock)
        pos_boundaryend_context_id.append(boundary.token_lock + 1)
        pos_boundaryend_types.append(1)
        pos_boundaryend_token_masks.append(create_entity_mask(*boundary.span_token, token_count))
        pos_boundaryend_context_masks.append(create_entity_mask(*boundary.span_context, context_size))



    neg_boundarystart_token_id, neg_boundarystart_context_id, neg_boundarystart_types,  neg_boundarystart_token_masks, neg_boundarystart_context_masks = [], [], [], [], []
    neg_boundaryend_token_id, neg_boundaryend_context_id, neg_boundaryend_types, neg_boundaryend_token_masks, neg_boundaryend_context_masks = [], [], [], [], []
    for i in range(token_count):
        if i not in pos_boundarystart_token_id:
            neg_boundarystart_token_id.append(i)
            neg_boundaryend_context_id.append(i+1)
            neg_boundarystart_types.append(0)
            neg_boundarystart_token_masks.append(create_entity_mask(doc.tokens[i].index, doc.tokens[i].index+1, token_count))
            neg_boundarystart_context_masks.append(create_entity_mask(doc.tokens[i].span_start, doc.tokens[i].span_start+1, context_size))
    for i in range(token_count):
        if i not in pos_boundaryend_token_id:
            neg_boundaryend_token_id.append(i)
            neg_boundaryend_context_id.append(i+1)
            neg_boundaryend_types.append(0)
            neg_boundaryend_token_masks.append(create_entity_mask(doc.tokens[i].index, doc.tokens[i].index+1, token_count))
            neg_boundaryend_context_masks.append(create_entity_mask(doc.tokens[i].span_start, doc.tokens[i].span_start+1, context_size))

    boundarystart_token_id=pos_boundarystart_token_id+neg_boundarystart_token_id
    boundarystart_context_id=pos_boundarystart_context_id+neg_boundaryend_context_id
    boundarystart_types = pos_boundarystart_types+neg_boundarystart_types
    boundarystart_token_masks=pos_boundarystart_token_masks+neg_boundarystart_token_masks
    boundarystart_context_masks = pos_boundaryend_context_masks+neg_boundarystart_context_masks

    boundaryend_token_id =pos_boundaryend_token_id+neg_boundaryend_token_id
    boundaryend_context_id = pos_boundaryend_context_id+neg_boundaryend_context_id
    boundaryend_types =pos_boundaryend_types+neg_boundaryend_types
    boundaryend_token_masks=pos_boundarystart_token_masks+neg_boundaryend_token_masks
    boundaryend_context_masks = pos_boundaryend_context_masks+neg_boundaryend_context_masks

    rankstart = [index for index, value in sorted(list(enumerate(boundarystart_token_id)), key=lambda x: x[1])]
    boundarystart_token_id = [boundarystart_token_id[i] for i in rankstart]
    boundarystart_context_id = [boundarystart_context_id[i] for i in rankstart]
    boundarystart_types = [boundarystart_types[i] for i in rankstart]
    boundarystart_token_masks = [boundarystart_token_masks[i] for i in rankstart]
    boundarystart_context_masks = [boundarystart_context_masks[i] for i in rankstart]

    rankend = [index for index, value in sorted(list(enumerate(boundaryend_token_id)), key=lambda x: x[1])]
    boundaryend_token_id = [boundaryend_token_id[i] for i in rankend]
    boundaryend_context_id = [boundaryend_context_id[i] for i in rankend]
    boundaryend_types = [boundaryend_types[i] for i in rankend]
    boundaryend_token_masks = [boundaryend_token_masks[i] for i in rankend]
    boundaryend_context_masks = [boundaryend_context_masks[i] for i in rankend]

    boundarystart_token_id = torch.tensor(boundarystart_token_id,dtype=torch.long)
    boundarystart_context_id = torch.tensor(boundarystart_context_id,dtype=torch.long)
    boundarystart_types = torch.tensor(boundarystart_types, dtype=torch.long)
    boundarystart_token_masks =torch.stack(boundarystart_token_masks)
    boundarystart_context_masks = torch.stack(boundarystart_context_masks)
    boundarystart_sample_mask =torch.ones([boundarystart_token_masks.shape[0]],dtype=torch.bool)

    boundaryend_token_id = torch.tensor(boundaryend_token_id, dtype=torch.long)
    boundaryend_context_id = torch.tensor(boundaryend_context_id, dtype=torch.long)
    boundaryend_types = torch.tensor(boundaryend_types, dtype=torch.long)
    boundaryend_token_masks = torch.stack(boundaryend_token_masks)
    boundaryend_context_masks = torch.stack(boundaryend_context_masks)
    boundaryend_sample_mask = torch.ones([boundaryend_token_masks.shape[0]], dtype=torch.bool)

    entities_span_token = []
    entities_span_context = []
    entities_span_types = []
    for e in doc.entities:
        entities_span_token.append(e.span_token)
        entities_span_context.append(e.span_context)
        entities_span_types.append(e.entity_type.index)
    entities_span_token_masks=[create_entity_mask(*span, token_count) for span in entities_span_token]
    entities_span_context_masks = [create_entity_mask(*span, context_size) for span in entities_span_context]

    entities_span_token = torch.tensor(entities_span_token,dtype=torch.long)
    entities_span_context = torch.tensor(entities_span_context,dtype=torch.long)
    entities_span_types = torch.tensor(entities_span_types,dtype=torch.long)
    entities_span_token_masks = torch.stack(entities_span_token_masks)
    entities_span_context_masks = torch.stack(entities_span_context_masks)
    entities_sample_masks = torch.ones([entities_span_token_masks.shape[0]],dtype=torch.bool)
    g_accusation_type=[]
    for a in doc.accusation:
        g_accusation_type.append(a.acc_type.index)
    #accusition_label=[0 for x in range(0,12)]
    accusation_type=[]
    for a in g_accusation_type:
        #accusition_label[a]=1
        accusation_type.append(a)

    encodings = torch.tensor(encodings, dtype=torch.long)
    context_masks = torch.ones(context_size, dtype=torch.bool)
    token_masks = torch.ones(token_count, dtype=torch.bool)
    accusation_type=torch.tensor(accusation_type, dtype=torch.long)

    return dict(encodings=encodings,context_masks=context_masks,token_masks=token_masks,
                entities_span_token=entities_span_token,entities_span_token_masks=entities_span_token_masks,entities_span_context=entities_span_context,entities_span_context_masks=entities_span_context_masks,
                entities_sample_masks=entities_sample_masks,entities_span_types=entities_span_types,
                boundarystart_token_id=boundarystart_token_id,boundarystart_context_id=boundarystart_context_id,boundarystart_types=boundarystart_types,
                boundarystart_sample_mask=boundarystart_sample_mask,boundarystart_token_masks=boundarystart_token_masks,boundarystart_context_masks=boundarystart_context_masks,
                boundaryend_token_id=boundaryend_token_id,boundaryend_context_id=boundaryend_context_id,boundaryend_types=boundaryend_types,
                boundaryend_sample_mask=boundaryend_sample_mask,boundaryend_token_masks=boundaryend_token_masks,boundaryend_context_masks=boundaryend_context_masks,
                accusation_type=accusation_type
                )

def create_eval_sample(doc):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    pos_boundarstart = doc.boundarystart
    pos_boundarend = doc.boundaryend
    pos_boundarystart_token_id, pos_boundarystart_context_id, pos_boundarystart_types, pos_entity_boundarystart_masks, pos_boundarystart_token_masks, pos_boundarystart_context_masks = [], [], [], [], [], []
    for boundary in pos_boundarstart:
        pos_boundarystart_token_id.append(boundary.token_lock)
        pos_boundarystart_context_id.append(boundary.token_lock + 1)
        pos_boundarystart_types.append(1)
        pos_boundarystart_token_masks.append(create_entity_mask(*boundary.span_token, token_count))
        pos_boundarystart_context_masks.append(create_entity_mask(*boundary.span_context, context_size))
    pos_boundaryend_token_id, pos_boundaryend_context_id, pos_boundaryend_types, pos_entity_boundaryend_masks, pos_boundaryend_token_masks, pos_boundaryend_context_masks = [], [], [], [], [], []
    for boundary in pos_boundarend:
        pos_boundaryend_token_id.append(boundary.token_lock)
        pos_boundaryend_context_id.append(boundary.token_lock + 1)
        pos_boundaryend_types.append(1)
        pos_boundaryend_token_masks.append(create_entity_mask(*boundary.span_token, token_count))
        pos_boundaryend_context_masks.append(create_entity_mask(*boundary.span_context, context_size))

    neg_boundarystart_token_id, neg_boundarystart_context_id, neg_boundarystart_types, neg_boundarystart_token_masks, neg_boundarystart_context_masks = [], [], [], [], []
    neg_boundaryend_token_id, neg_boundaryend_context_id, neg_boundaryend_types, neg_boundaryend_token_masks, neg_boundaryend_context_masks = [], [], [], [], []
    for i in range(token_count):
        if i not in pos_boundarystart_token_id:
            neg_boundarystart_token_id.append(i)
            neg_boundaryend_context_id.append(i + 1)
            neg_boundarystart_types.append(0)
            neg_boundarystart_token_masks.append(
                create_entity_mask(doc.tokens[i].index, doc.tokens[i].index + 1, token_count))
            neg_boundarystart_context_masks.append(
                create_entity_mask(doc.tokens[i].span_start, doc.tokens[i].span_start + 1, context_size))
    for i in range(token_count):
        if i not in pos_boundaryend_token_id:
            neg_boundaryend_token_id.append(i)
            neg_boundaryend_context_id.append(i + 1)
            neg_boundaryend_types.append(0)
            neg_boundaryend_token_masks.append(
                create_entity_mask(doc.tokens[i].index, doc.tokens[i].index + 1, token_count))
            neg_boundaryend_context_masks.append(
                create_entity_mask(doc.tokens[i].span_start, doc.tokens[i].span_start + 1, context_size))

    boundarystart_token_id = pos_boundarystart_token_id + neg_boundarystart_token_id
    boundarystart_context_id = pos_boundarystart_context_id + neg_boundaryend_context_id
    boundarystart_types = pos_boundarystart_types + neg_boundarystart_types
    boundarystart_token_masks = pos_boundarystart_token_masks + neg_boundarystart_token_masks
    boundarystart_context_masks = pos_boundaryend_context_masks + neg_boundarystart_context_masks

    boundaryend_token_id = pos_boundaryend_token_id + neg_boundaryend_token_id
    boundaryend_context_id = pos_boundaryend_context_id + neg_boundaryend_context_id
    boundaryend_types = pos_boundaryend_types + neg_boundaryend_types
    boundaryend_token_masks = pos_boundarystart_token_masks + neg_boundaryend_token_masks
    boundaryend_context_masks = pos_boundaryend_context_masks + neg_boundaryend_context_masks

    rankstart = [index for index, value in sorted(list(enumerate(boundarystart_token_id)), key=lambda x: x[1])]
    boundarystart_token_id = [boundarystart_token_id[i] for i in rankstart]
    boundarystart_context_id = [boundarystart_context_id[i] for i in rankstart]
    boundarystart_types = [boundarystart_types[i] for i in rankstart]
    boundarystart_token_masks = [boundarystart_token_masks[i] for i in rankstart]
    boundarystart_context_masks = [boundarystart_context_masks[i] for i in rankstart]

    rankend = [index for index, value in sorted(list(enumerate(boundaryend_token_id)), key=lambda x: x[1])]
    boundaryend_token_id = [boundaryend_token_id[i] for i in rankend]
    boundaryend_context_id = [boundaryend_context_id[i] for i in rankend]
    boundaryend_types = [boundaryend_types[i] for i in rankend]
    boundaryend_token_masks = [boundaryend_token_masks[i] for i in rankend]
    boundaryend_context_masks = [boundaryend_context_masks[i] for i in rankend]

    boundarystart_token_id = torch.tensor(boundarystart_token_id, dtype=torch.long)
    boundarystart_context_id = torch.tensor(boundarystart_context_id, dtype=torch.long)
    boundarystart_types = torch.tensor(boundarystart_types, dtype=torch.long)
    boundarystart_token_masks = torch.stack(boundarystart_token_masks)
    boundarystart_context_masks = torch.stack(boundarystart_context_masks)
    boundarystart_sample_mask = torch.ones([boundarystart_token_masks.shape[0]], dtype=torch.bool)

    boundaryend_token_id = torch.tensor(boundaryend_token_id, dtype=torch.long)
    boundaryend_context_id = torch.tensor(boundaryend_context_id, dtype=torch.long)
    boundaryend_types = torch.tensor(boundaryend_types, dtype=torch.long)
    boundaryend_token_masks = torch.stack(boundaryend_token_masks)
    boundaryend_context_masks = torch.stack(boundaryend_context_masks)
    boundaryend_sample_mask = torch.ones([boundaryend_token_masks.shape[0]], dtype=torch.bool)

    entities_span_token = []
    entities_span_context = []
    entities_span_types = []
    for e in doc.entities:
        entities_span_token.append(e.span_token)
        entities_span_context.append(e.span_context)
        entities_span_types.append(e.entity_type.index)
    entities_span_token_masks = [create_entity_mask(*span, token_count) for span in entities_span_token]
    entities_span_context_masks = [create_entity_mask(*span, context_size) for span in entities_span_context]

    entities_span_token = torch.tensor(entities_span_token, dtype=torch.long)
    entities_span_context = torch.tensor(entities_span_context, dtype=torch.long)
    entities_span_types = torch.tensor(entities_span_types, dtype=torch.long)
    entities_span_token_masks = torch.stack(entities_span_token_masks)
    entities_span_context_masks = torch.stack(entities_span_context_masks)
    entities_sample_masks = torch.ones([entities_span_token_masks.shape[0]], dtype=torch.bool)
    g_accusation_type = []
    for a in doc.accusation:
        g_accusation_type.append(a.acc_type.index)

    accusation_type = []
    for a in g_accusation_type:
        accusation_type.append(a)

    encodings = torch.tensor(encodings, dtype=torch.long)
    context_masks = torch.ones(context_size, dtype=torch.bool)
    token_masks = torch.ones(token_count, dtype=torch.bool)
    accusation_type = torch.tensor(accusation_type, dtype=torch.long)

    return dict(encodings=encodings, context_masks=context_masks, token_masks=token_masks,
                entities_span_token=entities_span_token, entities_span_token_masks=entities_span_token_masks,
                entities_span_context=entities_span_context, entities_span_context_masks=entities_span_context_masks,
                entities_sample_masks=entities_sample_masks, entities_span_types=entities_span_types,
                boundarystart_token_id=boundarystart_token_id, boundarystart_context_id=boundarystart_context_id,
                boundarystart_types=boundarystart_types,
                boundarystart_sample_mask=boundarystart_sample_mask,
                boundarystart_token_masks=boundarystart_token_masks,
                boundarystart_context_masks=boundarystart_context_masks,
                boundaryend_token_id=boundaryend_token_id, boundaryend_context_id=boundaryend_context_id,
                boundaryend_types=boundaryend_types,
                boundaryend_sample_mask=boundaryend_sample_mask, boundaryend_token_masks=boundaryend_token_masks,
                boundaryend_context_masks=boundaryend_context_masks,
                accusation_type=accusation_type
                )





def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch
