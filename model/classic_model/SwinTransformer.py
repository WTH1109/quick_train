import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm  # 需要安装 timm 库
from torchmetrics import Accuracy, ConfusionMatrix
from transformers import AutoModel, ViTFeatureExtractor, ViTModel


class SwinTransformerClassifier(pl.LightningModule):
    def __init__(self,
                 num_classes=2,
                 pretrained=True,
                 learning_rate=0.001,
                 model_name='swin_tiny_patch4_window7_224'):
        """
        Swin-Transformer 分类器

        :param num_classes: 类别数
        :param pretrained: 是否使用预训练权重
        :param learning_rate: 学习率
        :param model_name: Swin 模型名称（参考 timm 文档）
        """
        super().__init__()
        self.save_hyperparameters()

        # 初始化 Swin-Transformer 模型
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # 不要分类头，后面自定义
        )

        # self.model = ViTModel.from_pretrained("e1010101/vit-384-tongue-image")
        # 检查模型输出维度（假设 hidden_size=768）
        # hidden_size = self.model.config.hidden_size  # 通常为 768


        # # 获取特征维度
        num_features = self.model.num_features

        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 指标
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_conf_mat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        features = self.model(x)
        # last_hidden_state = features.last_hidden_state  # [batch, seq_len, hidden_size]
        #
        # # 取 [CLS] token 的特征（全局表示）
        # cls_token = last_hidden_state[:, 0, :]  # [batch, hidden_size]
        return self.classifier(features)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.05  # 通常 Transformer 需要更大的 weight decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=20, factor=0.9
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_acc"
        }

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        pre_class = torch.argmax(logits, dim=1)
        self.train_acc(pre_class, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        pre_class = torch.argmax(logits, dim=1)
        self.val_acc(pre_class, labels)
        self.val_conf_mat(pre_class, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

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

    def on_train_epoch_start(self):
        # 获取当前优化器
        optimizer = self.optimizers()
        # 遍历所有参数组（如多组学习率）
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            print(f"Epoch {self.current_epoch}: Optimizer {i}, lr = {lr}")



# 使用示例 --------------------------------------------------
if __name__ == "__main__":
    # 安装依赖：pip install timm pytorch_lightning
    model = SwinTransformerClassifier(
        num_classes=2,
        pretrained=True,
        learning_rate=2e-5,
        model_name='swin_small_patch4_window7_224'
    )
