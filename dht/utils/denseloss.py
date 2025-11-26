import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedTargets(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.register_buffer('_targets', torch.eye(n_classes), persistent=False)

    @staticmethod
    def compute_overlap(predseg, targetseg, maxseg:int):
        assert predseg.shape == targetseg.shape
        
        def uidxcnt(arr:torch.Tensor):
            '''Function to get unique indices and counts for a flat array.
            '''
            perm = arr.argsort()
            aux = arr[perm]
            mask = arr.new_zeros(aux.shape[0], dtype=torch.bool)
            mask[:1] = True
            mask[1:] = aux[1:] != aux[:-1]
            uidx = perm[mask]
            cnt = F.pad(mask.nonzero()[:,0], (0,1), value=mask.numel()).diff()
            return uidx, cnt

        # The idea here is that we want to construct a unique index for each mask in the batch.
        # To do this, we multiply the maximum number of segmentation classes by the batch index,
        # and add the target masks. This ensures that we get a full cover, and can retrieve the 
        # original labels using modulo operations at the end.        
        b_idx = torch.where(torch.ones_like(predseg, dtype=torch.bool))[0]  # Quick batch indices
        pred = predseg.view(-1)                                             # Flatten predictions to pixels
        target = b_idx * maxseg + targetseg.view(-1)                        # Give target mask in each image a unique index.
        ar = torch.stack([pred, target], 0)                                 # We now stack the predictions and targets
        m = ar.max() + 1                                                    # Find the maximum index in the stacked structure
        r, c = ar                                                           # We are just renaming r,c = pred, target
        cons = r*m + c                                                      # Flatten indices to do matching
        
        uidx_cons, n_overlap = uidxcnt(cons)
        _, inv_pred, cnt_pred = pred.unique(return_counts=True, return_inverse=True)
        _, inv_target, cnt_target = target.unique(return_counts=True, return_inverse=True)
        
        n_pred = cnt_pred[inv_pred[uidx_cons]]
        n_target = cnt_target[inv_target[uidx_cons]]
        
        ious = n_overlap / (n_pred + n_target - n_overlap)
    
        predidx, targetidx = ar[:, uidx_cons]
        targetidx = targetidx % maxseg
        
        return predidx, targetidx, ious

    def forward(self, predseg, targetseg):
        pidx, tidx, iou = self.compute_overlap(predseg, targetseg, self.n_classes)
        n = int(pidx.max().item()) + 1
        out = torch.zeros(n, self.n_classes, device=predseg.device)
        out.scatter_reduce_(0, pidx[:,None].expand(-1, self.n_classes), iou.view(-1,1) * self._targets[tidx], 'sum')
        den = out.sum(-1, keepdim=True)
        return out / den


class MaskedCrossEntropy(nn.Module):

    def __init__(self, n_classes, ignore_index=0, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        cls_weights = torch.ones(n_classes)
        cls_weights[ignore_index] = 0
        self.register_buffer('cls_weights', cls_weights)
        self.get_tgt = MaskedTargets(n_classes)

    def forward(self, pred, predseg, targetseg, argmax=False):
        target = self.get_tgt(predseg, targetseg)
        if argmax:
            target = target.argmax(-1)
        return F.cross_entropy(pred, target, reduction=self.reduction, weight=self.cls_weights)
        

class MaskedFocalLoss(nn.Module):

    def __init__(self, n_classes, gamma=4.0, ignore_index=0, normalize=True, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        cls_weights = torch.ones(n_classes)
        cls_weights[ignore_index] = 0
        self.register_buffer('cls_weights', cls_weights)
        self.get_tgt = MaskedTargets(n_classes)
        self._c = 1.0
        if normalize:
            self._c = self.normalization_factor().item()
            
    @staticmethod
    def harmonic_number(g, eps=1e-7):
        t = torch.linspace(0,1-eps,1000)
        return torch.trapz((1-t**g)/(1-t), t)

    def normalization_factor(self):
        return (self.gamma + 1) / self.harmonic_number(self.gamma + 1)

    def _comploss(self, pred, target):
        brng = torch.arange(pred.shape[0], device=pred.device)
        p = pred.softmax(-1)
        pt = p[brng,target.argmax(-1)]
        ce = F.cross_entropy(pred, target, reduction='none', weight=self.cls_weights)
        loss = (1 - pt)**self.gamma * ce * self._c
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()

    def forward(self, pred, predseg, targetseg):
        target = self.get_tgt(predseg, targetseg)
        brng = torch.arange(pred.shape[0], device=predseg.device)
        p = pred.softmax(-1)
        pt = p[brng,target.argmax(-1)]
        ce = F.cross_entropy(pred, target, reduction='none', weight=self.cls_weights)
        loss = (1 - pt)**self.gamma * ce * self._c
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        raise NotImplementedError(f'Reduction {self.reduction} not implemented.')


class MaskedMulticlassFromLabels(nn.Module):

    def __init__(
        self, n_classes:int, ignore_index:int=0, 
        label_smoothing:float=0.05, temperature:float=0.25, 
        bce:bool=True
    ):
        super().__init__()
        self.n_classes = n_classes
        self.get_tgt = MaskedTargets(n_classes)
        cls_weights = torch.ones(n_classes)
        cls_weights[ignore_index] = 0
        self.register_buffer('cls_weights', cls_weights)
        self.label_smoothing = label_smoothing
        self.temp = temperature
        self.bce = bce


    def forward(self, pred, targetseg):
        B,H,W = targetseg.shape
        bidx, cidx, scores = self.get_tgt.compute_overlap(
            torch.arange(B, device=targetseg.device)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(B,H,W)
                .contiguous(),
            targetseg,
            self.n_classes
        )
        target = F.normalize(
            torch.sparse_coo_tensor(torch.stack([bidx, cidx],0), scores, size=(B, self.n_classes))
                .to_dense()
                .pow(self.temp),
            p=1, dim=-1
        )
        
        if not self.bce:
            return F.cross_entropy(
                pred, target, weight=self.cls_weights, label_smoothing=self.label_smoothing
            )
        return F.binary_cross_entropy_with_logits(
            pred, target, weight=self.cls_weights
        )
    

class DenseLoss(nn.Module):

    def __init__(
        self, n_classes, ignore_index=0, label_smoothing=0.05, temperature=0.25, 
        gamma=4.0, normalize=True, reduction='mean', num_cls_tokens=1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.temp = temperature
        self.gamma = gamma
        self.normalize = normalize
        self.reduction = reduction
        self.num_cls_tokens = num_cls_tokens

        self.dense_loss = MaskedFocalLoss(n_classes, gamma, ignore_index, normalize, reduction)
        self.image_loss = MaskedMulticlassFromLabels(n_classes, ignore_index, label_smoothing, temperature)

    def forward(self, predicion_tuple, labels):
        pred, seg, amask = predicion_tuple
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        nc = self.num_cls_tokens
        image_loss = self.image_loss(pred[:,0], labels)
        dense_loss = self.dense_loss(pred[:,nc:][amask[:,nc:]], seg, labels)
        return image_loss, dense_loss

