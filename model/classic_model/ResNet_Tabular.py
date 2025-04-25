import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import ConfusionMatrix, Accuracy
from pytorch_lightning.callbacks import TQDMProgressBar


class FusionModel(pl.LightningModule):
    def __init__(self,
                 num_classes=2,
                 struct_feature_dim=10,  # 假设指标数据维度为10
                 pretrained=True,
                 learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        # 图像特征提取
        self.img_backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        num_img_features = self.img_backbone.fc.in_features
        self.img_backbone.fc = nn.Identity()  # 移除原全连接层
        self.weights = nn.Parameter(torch.ones(struct_feature_dim))

        # 结构化数据处理
        self.struct_fc = nn.Sequential(
            nn.Linear(struct_feature_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
        )

        self.image_proj = nn.Sequential(
            nn.Linear(num_img_features, 64),
            nn.ReLU(),
        )

        # 融合分类器
        self.classifier = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        self.balance_weights = nn.Parameter(torch.ones(64 + 64))

        # 初始化指标
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_conf_mat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, img, struct):
        # 提取图像特征
        img_features = self.img_backbone(img)
        img_features= self.image_proj(img_features)

        struct = self.weights * struct
        # 处理结构化数据
        struct_features = self.struct_fc(struct)

        # 特征融合
        combined = torch.cat([img_features, struct_features], dim=1)
        combined = self.balance_weights * combined
        return self.classifier(combined)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5),
                "monitor": "val_acc",
                "interval": "epoch"
            }
        }

    def training_step(self, batch, batch_idx):
        img, struct, labels = batch  # 现在batch包含三个元素

        outputs = self(img, struct)
        loss = self.criterion(outputs, labels)

        self.train_acc(outputs, labels)
        self.log_dict({
            "train_loss": loss,
            "train_acc": self.train_acc
        }, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        # 获取当前优化器
        optimizer = self.optimizers()
        # 遍历所有参数组（如多组学习率）
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            print(f"Epoch {self.current_epoch}: Optimizer {i}, lr = {lr}")

    def validation_step(self, batch, batch_idx):
        img, struct, labels = batch
        outputs = self(img, struct)
        loss = self.criterion(outputs, labels)

        pre_class = torch.argmax(outputs, dim=1)
        self.val_conf_mat(pre_class, labels)

        # 更新验证准确度
        self.val_acc(outputs, labels)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc)
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


class TabularModel(pl.LightningModule):
    def __init__(self,
                 num_classes=2,
                 struct_feature_dim=10,  # 假设指标数据维度为10
                 learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.weights = nn.Parameter(torch.ones(struct_feature_dim))
        # 结构化数据处理
        self.struct_fc = nn.Sequential(
            nn.Linear(struct_feature_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
        )

        # 融合分类器
        self.classifier = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # 初始化指标
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_conf_mat = ConfusionMatrix(task="binary", num_classes=2)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, img, struct):
        # 处理结构化数据
        struct = self.weights * struct
        struct_features = self.struct_fc(struct)
        return self.classifier(struct_features)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5),
                "monitor": "val_acc",
                "interval": "epoch"
            }
        }

    def on_train_epoch_start(self):
        # 获取当前优化器
        optimizer = self.optimizers()
        # 遍历所有参数组（如多组学习率）
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            print(f"Epoch {self.current_epoch}: Optimizer {i}, lr = {lr}")

    def training_step(self, batch, batch_idx):
        img, struct, labels = batch  # 现在batch包含三个元素
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2)

        outputs = self(img, struct)
        loss = self.criterion(outputs, labels)

        self.train_acc(outputs, one_hot_labels)
        self.log_dict({
            "train_loss": loss,
            "train_acc": self.train_acc
        }, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, struct, labels = batch
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2)
        outputs = self(img, struct)
        loss = self.criterion(outputs, labels)

        pre_class = torch.argmax(outputs, dim=1)
        self.val_conf_mat(pre_class, labels)

        # 更新验证准确度
        self.val_acc(outputs, one_hot_labels)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc)
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