import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torchmetrics.classification import Accuracy

from utils.config_utils import instantiate_from_config


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes, pretrained=True, learning_rate=0.001):
        """
        初始化 ResNet 分类器

        :param num_classes: 类别数
        :param pretrained: 是否使用预训练的权重
        :param learning_rate: 学习率
        """
        super(ResNetClassifier, self).__init__()
        # 初始化 ResNet 模型
        if pretrained:
            self.model = models.resnet18(weights="IMAGENET1K_V1")
        else:
            self.model = models.resnet18()
        # 获取 ResNet 最后一层的输入维度
        num_feature_last = self.model.fc.in_features
        # 修改全连接层的输出维度
        self.model.fc = nn.Linear(num_feature_last, num_classes)
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        # 指定学习率
        self.learning_rate = float(learning_rate)
        # 初始化准确度度量
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")

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
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_acc"}


    def training_step(self, batch, batch_idx):
        """
        训练步骤

        :param batch: 输入数据和标签
        :param batch_idx: 批次索引
        :return: 损失
        """
        images, labels = batch
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2)
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # 更新训练准确度
        self.train_acc(outputs, one_hot_labels)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        验证步骤

        :param batch: 输入数据和标签
        :param batch_idx: 批次索引
        :return: 损失
        """
        images, labels = batch
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2)
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # 更新验证准确度
        self.val_acc(outputs, one_hot_labels)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc, prog_bar=True)

        return loss