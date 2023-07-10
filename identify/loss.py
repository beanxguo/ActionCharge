import torch

class IdentifierLoss():
    def __init__(self,entity_loss,boundarystart_loss,boundaryend_loss,accusition_criterion,model,optimizer,scheduler,max_grad_norm):
        self._entity_loss = entity_loss
        self._boundarystart_loss = boundarystart_loss
        self._boundaryend_loss = boundaryend_loss
        self._addusition_criterion = accusition_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm=max_grad_norm
    def compute(self,entity_clf,entity_span_type,entity_span_sample_mask,
                    bs_cls,boundarystart_types,boundarystart_sample_mask,
                    be_cls,boundaryend_types,boundaryend_sample_mask
                    ,pre_acc_node,pre_acc_node_ample,gt_accusation):
        pre_entity_type=entity_clf.view(-1, entity_clf.shape[-1])
        gt_entity_type=entity_span_type.view(-1)
        entity_sample=entity_span_sample_mask.view(-1).float()
        entity_loss = self._entity_loss(pre_entity_type,gt_entity_type)
        entity_loss = (entity_loss*entity_sample).sum()/entity_sample.sum()

        pre_bs_type=bs_cls.view(-1,bs_cls.shape[-1])
        gt_boundarystart=boundarystart_types.view(-1)
        boundarystart_sample = boundarystart_sample_mask.view(-1).float()
        boundarystart_loss = self._boundarystart_loss(pre_bs_type,gt_boundarystart)
        boundarystart_loss=(boundarystart_loss*boundarystart_sample).sum()/boundarystart_sample.sum()

        be_cls = be_cls.view(-1, be_cls.shape[-1])
        gt_boundaryend = boundaryend_types.view(-1)
        boundaryend_sample = boundaryend_sample_mask.view(-1).float()
        boundaryend_loss = self._boundaryend_loss(be_cls, gt_boundaryend)
        boundaryend_loss = (boundaryend_loss * boundaryend_sample).sum() / boundaryend_sample.sum()

        gt_accusation = gt_accusation.unsqueeze(1).repeat(1, pre_acc_node.shape[1], 1)
        pre_acc_node=pre_acc_node.view(-1,pre_acc_node.shape[-1])
        gt_accusation=gt_accusation.view(-1)
        pre_acc_node_ample=pre_acc_node_ample.view(-1).float()
        accusation_loss=self._addusition_criterion(pre_acc_node,gt_accusation)
        accusation_loss=(accusation_loss*pre_acc_node_ample).sum() / pre_acc_node_ample.sum()


        train_loss=accusation_loss+entity_loss+60*boundarystart_loss+60*boundaryend_loss
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()



