import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts, \
    LinearLR, SequentialLR
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import Accuracy

from utils.config_utils import instantiate_from_config


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes, pretrained=True, model='resnet18', learning_rate=0.001, drop_out=0.5):
        """
        初始化 ResNet 分类器

        :param num_classes: 类别数
        :param pretrained: 是否使用预训练的权重
        :param learning_rate: 学习率
        """
        super(ResNetClassifier, self).__init__()
        self.num_classes = num_classes
        # 初始化 ResNet 模型
        if pretrained:
            if model == 'resnet18':
                self.model = models.resnet18(weights="IMAGENET1K_V1")
            elif model == 'resnet34':
                self.model = models.resnet34(weights="IMAGENET1K_V1")
            elif model == 'resnet50':
                self.model = models.resnet50(weights="IMAGENET1K_V1")
            elif model == 'resnet101':
                self.model = models.resnet101(weights="IMAGENET1K_V1")
            elif model.startswith('efficientnet'):
                if model == 'efficientnet_b0':
                    self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")
                elif model == 'efficientnet_b4':
                    self.model = models.efficientnet_b4(weights="IMAGENET1K_V1")
                # 补充Vision Transformer
            elif model.startswith('vit'):
                if model == 'vit_b_16':
                    self.model = models.vit_b_16(weights="IMAGENET1K_V1")
                elif model == 'vit_l_16':
                    self.model = models.vit_l_16(weights="IMAGENET1K_SAM")
                # 补充Inception
            elif model == 'inception_v3':
                self.model = models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)  # 禁用辅助分类器
                self.model.aux_logits = False  # 禁用辅助分类器输出
                self.model.AuxLogits = None  # 彻底移除辅助分类器层
        else:
            if model == 'resnet18':
                self.model = models.resnet18()
            elif model == 'resnet34':
                self.model = models.resnet34()
            elif model == 'resnet50':
                self.model = models.resnet50()
            elif model == 'resnet101':
                self.model = models.resnet101()
            elif model.startswith('efficientnet'):
                if model == 'efficientnet_b0':
                    self.model = models.efficientnet_b0()
                elif model == 'efficientnet_b4':
                    self.model = models.efficientnet_b4()
                # 补充Vision Transformer
            elif model.startswith('vit'):
                if model == 'vit_b_16':
                    self.model = models.vit_b_16()
                elif model == 'vit_l_16':
                    self.model = models.vit_l_16()
                # 补充Inception
            elif model == 'inception_v3':
                self.model = models.inception_v3(aux_logits=False)  # 禁用辅助分类器

        if 'resnet' in model:
            num_feature_last = self.model.fc.in_features
            self.model.fc = self._make_classifier(num_feature_last, num_classes, drop_out)
        elif 'efficientnet' in model:
            # EfficientNet的classifier结构不同
            num_feature_last = self.model.classifier[1].in_features
            self.model.classifier = self._make_classifier(num_feature_last, num_classes, drop_out)
        elif 'vit' in model:
            # ViT的head结构特殊
            num_feature_last = self.model.hidden_dim  # vit_b_16的hidden_dim=768
            self.model.heads.head = self._make_classifier(num_feature_last, num_classes, drop_out)
        elif 'inception' in model:
            # Inception_v3主分类器
            num_feature_last = self.model.fc.in_features
            self.model.fc = self._make_classifier(num_feature_last, num_classes, drop_out)
            # 如果需要处理辅助分类器（通常不需要）：
            # self.model.AuxLogits.fc = ...

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        # 指定学习率
        self.learning_rate = float(learning_rate)
        # 初始化准确度度量
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_conf_mat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.val_sensitivity = torchmetrics.Recall(task='multiclass', average='macro', num_classes=num_classes)
        self.val_specificity = torchmetrics.Specificity(task='multiclass', average='macro', num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task='multiclass', average='macro', num_classes=num_classes)
        self.val_auc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes, average='macro',validate_args=True)

        self.best_acc = 0
        self.best_epoch = 0
        self.best_info = ''

    def _make_classifier(self, in_features, num_classes, dropout_rate):
        """通用分类器构建方法"""
        return nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 64),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def forward(self, x):
        """
        前向传播

        :param x: 输入数据
        :return: 模型输出
        """
        return self.model(x)

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器

        :return: 优化器
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)
        # 参数设置
        # total_epochs = self.trainer.max_epochs  # 总训练 epoch 数
        # warmup_epochs = total_epochs // 100  # Warmup 阶段 epoch 数
        # eta_min = self.learning_rate * 0.01  # 学习率最小值
        # # 定义 Warmup 调度器
        # warmup = LinearLR(
        #     optimizer,
        #     start_factor=0.01,  # 初始学习率 = 0.001 * 0.01 = 1e-5
        #     end_factor=1.0,  # Warmup 结束后恢复基础学习率 0.001
        #     total_iters=warmup_epochs
        # )
        # # 定义余弦退火调度器（无重启）
        # cosine_scheduler = CosineAnnealingLR(
        #     optimizer,
        #     T_max=total_epochs - warmup_epochs,  # 关键：总 epoch 数减去 Warmup
        #     eta_min=eta_min
        # )

        reduce_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',  # 监控 val_loss 应选 'min'，监控 val_acc 选 'max'
            factor=0.9,  # 衰减系数
            patience=30,  # 容忍5个epoch不改善
            verbose=True  # 打印调整日志
        )

        # # 组合调度器（先 Warmup，后余弦退火）
        # scheduler = SequentialLR(
        #     optimizer,
        #     schedulers=[warmup, reduce_scheduler],
        #     milestones=[warmup_epochs]  # Warmup 结束后切换
        # )

        return {"optimizer": optimizer, "lr_scheduler": reduce_scheduler, "monitor": "val_acc"}


    def training_step(self, batch, batch_idx):
        """
        训练步骤

        :param batch: 输入数据和标签
        :param batch_idx: 批次索引
        :return: 损失
        """
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # 更新训练准确度
        self.train_acc(outputs, labels)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        验证步骤

        :param batch: 输入数据和标签
        :param batch_idx: 批次索引
        :return: 损失
        """
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        probs = probs.reshape(-1, self.num_classes)  # 强制保持二维结构


        pre_class = torch.argmax(probs, dim=1)
        self.val_conf_mat(pre_class, labels)
        self.val_sensitivity(pre_class, labels)
        self.val_specificity(pre_class, labels)
        self.val_f1(pre_class, labels)
        self.val_auc.update(probs, labels)

        # 更新验证准确度
        self.val_acc(outputs, labels)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc)

        return loss

    def on_train_epoch_start(self):
        # 获取当前优化器
        optimizer = self.optimizers()
        # 遍历所有参数组（如多组学习率）
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            print(f"Epoch {self.current_epoch}: Optimizer {i}, lr = {lr}")

    def on_validation_epoch_end(self):
        # 获取并打印混淆矩阵
        val_conf_mat = self.val_conf_mat.compute().cpu().numpy()
        # 获取当前 epoch 的训练和验证准确率
        train_acc_epoch = self.train_acc.compute() * 100
        val_acc_epoch = self.val_acc.compute() * 100

        val_sensitivity_epoch = self.val_sensitivity.compute().item() * 100
        val_specificity_epoch = self.val_specificity.compute().item() * 100
        val_f1_epoch = self.val_f1.compute().item() * 100
        val_auc_epoch = self.val_auc.compute().item() * 100

        print('\n')
        print('********************************************************************************')
        # 打印结果
        print(f"Epoch {self.current_epoch}:")
        print(f"Temp epoch {self.current_epoch} Train Acc={train_acc_epoch:.2f} Val Acc={val_acc_epoch:.2f} Sen={val_sensitivity_epoch:.2f} Spec={val_specificity_epoch:.2f} F1={val_f1_epoch:.2f} AUC={val_auc_epoch:.2f}")
        print(val_conf_mat)
        print('********************************************************************************')


        if val_acc_epoch > self.best_acc:
            self.best_acc = val_acc_epoch
            self.best_epoch = self.current_epoch
            self.best_info = f'Best epoch:{self.best_epoch} Train Acc={train_acc_epoch:.2f}; Val Acc={val_acc_epoch:.2f} Sen={val_sensitivity_epoch:.2f} Spec={val_specificity_epoch:.2f} F1={val_f1_epoch:.2f} AUC={val_auc_epoch:.2f}'
        print(self.best_info)
        print('********************************************************************************')
        print('')

        # 重置指标（可选）
        self.val_conf_mat.reset()
        self.train_acc.reset()
        self.val_acc.reset()
        self.val_sensitivity.reset()
        self.val_specificity.reset()
        self.val_f1.reset()
        self.val_auc.reset()

