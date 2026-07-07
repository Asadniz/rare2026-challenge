"""Model definitions and loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import timm
import logging
from peft import LoraConfig, get_peft_model
from collections import OrderedDict
from transformers import AutoModel
from .utils import log_print

logger = logging.getLogger("training.models")

def _process_state_dict(checkpoint):
    """Normalize checkpoint format so it can always be loaded into timm ResNet."""
    # Case 1: full checkpoint with "state_dict"
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Strip "student_backbone." prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("student_backbone."):
            new_key = k.replace("student_backbone.", "", 1)
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    return new_state_dict

class ViTLoRAClassifier(nn.Module):
    def __init__(self, num_classes=2, model_name="vit_base_patch16_224", lora_rank=8):
        super().__init__()
        
        backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=16,
            target_modules=["qkv"],
            lora_dropout=0.1,
        )
        
        self.backbone = get_peft_model(backbone, lora_config)
        hidden_dim = backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.backbone.print_trainable_parameters()
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_parameter_groups(self, base_lr, backbone_lr_multiplier=0.1):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.classifier.parameters())
        return [
            {"params": backbone_params, "lr": base_lr * backbone_lr_multiplier},
            {"params": head_params, "lr": base_lr},
        ]


class EfficientNetV2Classifier(nn.Module):
    def __init__(self, num_classes=2, model_name="tf_efficientnetv2_s"):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        hidden_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"EfficientNetV2: {total/1e6:.1f}M total params")
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_parameter_groups(self, base_lr, backbone_lr_multiplier=0.1):
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.classifier.parameters())
        return [
            {"params": backbone_params, "lr": base_lr * backbone_lr_multiplier},
            {"params": head_params, "lr": base_lr},
        ]


class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=2, model_name="convnext_tiny"):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        hidden_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"ConvNeXt: {total/1e6:.1f}M total params")
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_parameter_groups(self, base_lr, backbone_lr_multiplier=0.1):
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.classifier.parameters())
        return [
            {"params": backbone_params, "lr": base_lr * backbone_lr_multiplier},
            {"params": head_params, "lr": base_lr},
        ]

class DINOv3Classifier(nn.Module):
    def __init__(self, num_classes=2, unfreeze_blocks=2, huggingface_cache_dir=None):
        super().__init__()

        log_print("DINOv3: downloading/loading facebook/dinov3-vith16plus-pretrain-lvd1689m (may take a few minutes)...")
        self.backbone = AutoModel.from_pretrained(
            "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            cache_dir=huggingface_cache_dir
        )
        log_print("DINOv3: backbone loaded from HuggingFace")
        
        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last N transformer blocks
        for param in self.backbone.layer[-unfreeze_blocks:].parameters():
            param.requires_grad = True
        
        # Unfreeze final norm
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
        
        # Classification head on top of CLS token (hidden dim is 1280 for ViT-H)
        hidden_dim = self.backbone.config.hidden_size  # 1280
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        log_print(f"DINOv3: trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M params ({100*trainable/total:.1f}%), "
                  f"unfreeze_blocks={unfreeze_blocks}")
        logger.info(f"DINOv3: Trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M params ({100*trainable/total:.1f}%)")
    
    def forward(self, x):
        outputs = self.backbone(x)
        cls_token = outputs.last_hidden_state[:, 0]  # CLS token
        return self.classifier(cls_token)
    
    def get_parameter_groups(self, base_lr, backbone_lr_multiplier=0.1):
        """Return parameter groups with differential learning rates."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.classifier.parameters())
        return [
            {"params": backbone_params, "lr": base_lr * backbone_lr_multiplier},
            {"params": head_params, "lr": base_lr},
        ]

class ResNet50GastroNet(nn.Module):
    """ResNet50 model with GastroNet (https://doi.org/10.1016/j.media.2024.103298) weights adaptation."""

    def __init__(
        self,
        num_classes: int = 2,
        repo_id: str = "tgwboers/GastroNet-5M_Pretrained_Weights",
        filename: str = "RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1.pth",
        our_weights: bool = False
    ):
        super(ResNet50GastroNet, self).__init__()

        # Load model with GastroNet weights
        try:
            # First create a standard ResNet50
            self.backbone = timm.create_model("resnet50", pretrained=False, num_classes=num_classes)

            if our_weights:
                model_path = filename
                checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
            else:
                # Then download and load the specific GastroNet weights
                model_path = hf_hub_download(repo_id=repo_id, filename=filename)

                # Load the weights
                checkpoint = torch.load(model_path, map_location="cpu")

            state_dict = _process_state_dict(checkpoint)
            self.backbone.load_state_dict(state_dict, strict=False)

            logger.info(f"Successfully loaded GastroNet weights from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load GastroNet weights: {e}")
            # Fallback to ImageNet pretrained
            self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)

        # Ensure correct number of classes
        if hasattr(self.backbone, "fc"):
            if self.backbone.fc.out_features != num_classes:
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif hasattr(self.backbone, "classifier"):
            if self.backbone.classifier.out_features != num_classes:
                self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class CE_PPVAtRecallLoss(nn.Module):
    """
    CE + ranking for PPV at fixed recall level (binary model with 2 logits).
    """

    def __init__(self, recall_level=0.90, lambda_=1.0, margin=0.5, beta=0.99, class_weights=None):
        super(CE_PPVAtRecallLoss, self).__init__()
        if not (0 < recall_level < 1):
            raise ValueError("recall_level must be between 0 and 1.")

        self.recall_level = recall_level
        self.quantile = 1.0 - recall_level
        self.lambda_ = lambda_
        self.margin = margin
        self.beta = beta
        self.class_weights = class_weights

        self.register_buffer("ema_threshold", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def forward(self, inputs, targets):
        # Ensure long targets
        targets = targets.long()

        base_loss = F.cross_entropy(inputs, targets, weight=self.class_weights)

        # logits for positive class
        pos_logits = inputs[:, 1]
        pos_sample_logits = pos_logits[targets == 1]
        neg_sample_logits = pos_logits[targets == 0]

        if pos_sample_logits.numel() == 0 or neg_sample_logits.numel() == 0:
            return base_loss

        with torch.no_grad():
            batch_threshold = torch.quantile(pos_sample_logits, self.quantile)
            if not self.initialized:
                self.ema_threshold.copy_(batch_threshold)
                self.initialized = torch.tensor(True, device=self.ema_threshold.device)
            else:
                self.ema_threshold.mul_(self.beta).add_(batch_threshold, alpha=1 - self.beta)

        violations = neg_sample_logits - (self.ema_threshold.detach() - self.margin)
        ranking_loss = torch.mean(torch.relu(violations))
        return base_loss + self.lambda_ * ranking_loss


class DifferentiableSurrogateLoss(nn.Module):
    """
    Differentiable surrogate loss optimizing Precision (PPV) with recall constraint for binary case.
    """

    def __init__(self, recall_threshold=0.9, lambda_penalty=1.0, epsilon=1e-8):
        super(DifferentiableSurrogateLoss, self).__init__()
        self.recall_threshold = recall_threshold
        self.lambda_penalty = lambda_penalty
        self.epsilon = epsilon

    def forward(self, logits, targets):
        if logits.dim() > 1:
            logits = logits.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()

        y = targets.float()
        probs = torch.sigmoid(logits)

        tp_soft = y * probs
        fp_soft = (1 - y) * probs
        fn_soft = y * (1 - probs)

        tp_sum = torch.sum(tp_soft)
        fp_sum = torch.sum(fp_soft)
        fn_sum = torch.sum(fn_soft)

        precision = tp_sum / (tp_sum + fp_sum + self.epsilon)
        recall = tp_sum / (tp_sum + fn_sum + self.epsilon)

        recall_penalty = F.relu(self.recall_threshold - recall)
        loss = -precision + self.lambda_penalty * recall_penalty
        return loss


class MultiClassSurrogateLoss(nn.Module):
    """
    Multi-class version using softmax probabilities.
    """

    def __init__(self, num_classes, recall_threshold=0.9, lambda_penalty=1.0, epsilon=1e-8, reduction="mean"):
        super(MultiClassSurrogateLoss, self).__init__()
        self.num_classes = num_classes
        self.recall_threshold = recall_threshold
        self.lambda_penalty = lambda_penalty
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()

        total_loss = 0.0
        for c in range(self.num_classes):
            y_c = targets_onehot[:, c]
            probs_c = probs[:, c]

            tp_soft = y_c * probs_c
            fp_soft = (1 - y_c) * probs_c
            fn_soft = y_c * (1 - probs_c)

            tp_sum = torch.sum(tp_soft)
            fp_sum = torch.sum(fp_soft)
            fn_sum = torch.sum(fn_soft)

            precision_c = tp_sum / (tp_sum + fp_sum + self.epsilon)
            recall_c = tp_sum / (tp_sum + fn_sum + self.epsilon)

            recall_penalty_c = F.relu(self.recall_threshold - recall_c)
            loss_c = -precision_c + self.lambda_penalty * recall_penalty_c

            if self.reduction == "mean":
                total_loss += loss_c / self.num_classes
            else:
                total_loss += loss_c

        return total_loss

def create_loss_function(config, class_weights=None):
    """Create loss function based on configuration."""
    weights = class_weights if config.use_class_weights else None
    if weights is not None and isinstance(weights, torch.Tensor):
        # ensure device set later in forward
        pass

    lt = config.loss_type

    if lt == "ppv":
        return CE_PPVAtRecallLoss(
            recall_level=0.9,
            lambda_=config.ppv_lambda,
            margin=config.ppv_margin,
            beta=config.ppv_beta,
            class_weights=weights,
        )
    if lt == "surrogate":
        if config.num_classes == 1:
            return DifferentiableSurrogateLoss(lambda_penalty=config.surrogate_lambda)
        else:
            return MultiClassSurrogateLoss(num_classes=config.num_classes, lambda_penalty=config.surrogate_lambda)

    # default
    return nn.CrossEntropyLoss(weight=weights)


def create_model(config, phase="train"):
    num_classes = config.num_classes

    if config.model_type == "resnet50":
        if config.model_filename is not None:
            return ResNet50GastroNet(num_classes=num_classes, filename=config.model_filename, our_weights=config.our_weights)
        else:
            return ResNet50GastroNet(num_classes=num_classes, our_weights=config.our_weights)
    
    elif config.model_type == "dinov3":
        return DINOv3Classifier(
            num_classes=num_classes,
            unfreeze_blocks=getattr(config, 'dinov3_unfreeze_blocks', 2),
            huggingface_cache_dir=getattr(config, 'huggingface_cache_dir', None)
        )
    
    elif config.model_type == "vit_lora":
        return ViTLoRAClassifier(
            num_classes=num_classes,
            lora_rank=getattr(config, 'lora_rank', 8)
        )
    
    elif config.model_type == "efficientnetv2":
        return EfficientNetV2Classifier(num_classes=num_classes)
    
    elif config.model_type == "convnext":
        return ConvNeXtClassifier(num_classes=num_classes)
    
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")