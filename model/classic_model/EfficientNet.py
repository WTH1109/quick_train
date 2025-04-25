import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import Accuracy

from utils.config_utils import instantiate_from_config


class EfficientClassifier(pl.LightningModule):
    def __init__(self, num_classes, pretrained=True, learning_rate=0.001):
        """
        初始化 ResNet 分类器

        :param num_classes: 类别数
        :param pretrained: 是否使用预训练的权重
        :param learning_rate: 学习率
        """
        super(EfficientClassifier, self).__init__()
        # 初始化 ResNet 模型
        if pretrained:
            self.model = models.efficientnet_b3(weights="EfficientNet_B3_Weights.DEFAULT")
        else:
            self.model = models.efficientnet_b3()
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        # 指定学习率
        self.learning_rate = float(learning_rate)
        # 初始化准确度度量
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_conf_mat = ConfusionMatrix(task="multiclass", num_classes=num_classes)


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
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_acc"}


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

        pre_class = torch.argmax(outputs, dim=1)
        self.val_conf_mat(pre_class, labels)

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
        train_acc_epoch = self.train_acc.compute()
        val_acc_epoch = self.val_acc.compute()

        print('')
        print('****************************************')
        # 打印结果
        print(f"Epoch {self.current_epoch}:")
        print(f"Train Accuracy = {train_acc_epoch:.4f}")
        print(f"Val Accuracy = {val_acc_epoch:.4f}")
        print("Validation Confusion Matrix:")
        print(val_conf_mat)
        print('****************************************')

        # 重置指标（可选）
        self.val_conf_mat.reset()
        self.train_acc.reset()
        self.val_acc.reset()