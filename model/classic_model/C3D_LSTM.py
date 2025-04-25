import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
from torchmetrics import Accuracy
from torchmetrics.classification import MultilabelAccuracy, BinaryAccuracy


class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, pretrained=False):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc6 = nn.Linear(8192, 4096)
        # self.fc7 = nn.Linear(4096, 4096)
        # self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        # print(x.shape)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = self.dropout(x)
        b, c, d, h, w = x.size()

        x = self.adaptive_pool(x.view(-1, h, w)) # (B * C * T, H, W) -> (B * C * T, 1, 1)
        x = x.view(b, c, d)
        x = x.transpose(1, 2)
        # print(x.shape)
        return x

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
            # Conv1
            "features.0.weight": "conv1.weight",
            "features.0.bias": "conv1.bias",
            # Conv2
            "features.3.weight": "conv2.weight",
            "features.3.bias": "conv2.bias",
            # Conv3a
            "features.6.weight": "conv3a.weight",
            "features.6.bias": "conv3a.bias",
            # Conv3b
            "features.8.weight": "conv3b.weight",
            "features.8.bias": "conv3b.bias",
            # Conv4a
            "features.11.weight": "conv4a.weight",
            "features.11.bias": "conv4a.bias",
            # Conv4b
            "features.13.weight": "conv4b.weight",
            "features.13.bias": "conv4b.bias",
            # Conv5a
            "features.16.weight": "conv5a.weight",
            "features.16.bias": "conv5a.bias",
            # Conv5b
            "features.18.weight": "conv5b.weight",
            "features.18.bias": "conv5b.bias",
            # fc6
            "classifier.0.weight": "fc6.weight",
            "classifier.0.bias": "fc6.bias",
            # fc7
            "classifier.3.weight": "fc7.weight",
            "classifier.3.bias": "fc7.bias",
        }

        p_dict = torch.load('/mnt/Model/c3d/c3d-pretrained.pth')
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class C3DLstm(nn.Module):
    def __init__(self, num_classes=2, feature_dim=512, hidden_dim=256, num_layers=1, pool_type='max'):
        super(C3DLstm, self).__init__()
        self.c3d = C3D(num_classes=feature_dim)
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)  # 输出逐帧预测值
        self.pool_type = pool_type


    def forward(self, x):
        # x: (B, C, T, H, W)
        batch_size, channels, total_frames, height, width = x.size()

        clip_length = 16
        step = 8  # 分片的滑动步长
        clip_features = []

        for i in range(0, total_frames - clip_length + 1, step):
            clip = x[:, :, i:i+clip_length, :, :]  # 取连续 clip_length 帧
            feature = self.c3d(clip)  # (B, D, feature_dim)
            clip_features.append(feature)

        # 拼接所有 clip 的特征
        clip_features = torch.cat(clip_features, dim=1)  # (B, T_clips, feature_dim)

        # 输入 LSTM
        lstm_out, _ = self.lstm(clip_features)  # (B, T_clips, hidden_dim)
        predictions = self.fc(lstm_out)  # (B, T_clips, num_class)

        # 对 T_clips 维度进行聚合
        if self.pool_type == 'average':
            predictions = predictions.mean(dim=1)  # 平均池化
        elif self.pool_type == 'max':
            predictions = predictions.max(dim=1).values  # 最大池化
        elif self.pool_type == 'weighted':
            weights = torch.softmax(predictions.mean(dim=-1), dim=1)  # 动态生成权重
            predictions = (predictions * weights.unsqueeze(-1)).sum(dim=1)  # 加权求和
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        return predictions


class C3DLstmLighting(pl.LightningModule):
    def __init__(self, num_classes=2, feature_dim=512, hidden_dim=256, num_layers=1, pool_type='max', learning_rate=0.0001):
        super(C3DLstmLighting, self).__init__()
        self.c3d_lstm = C3DLstm(num_classes=num_classes, feature_dim=feature_dim, num_layers=num_layers,
                                hidden_dim=hidden_dim, pool_type=pool_type)
        self.loss_fn = nn.BCEWithLogitsLoss()  # 多标签分类任务
        # 指定学习率
        self.learning_rate = float(learning_rate)

        # 定义准确率指标
        self.train_acc = MultilabelAccuracy(num_labels=num_classes, threshold=0.5, average='micro')
        self.test_acc = MultilabelAccuracy(num_labels=num_classes, threshold=0.5, average='micro')

        self.train_class_acc = BinaryAccuracy(threshold=0.5)
        self.test_class_acc = BinaryAccuracy(threshold=0.5)

        self.train_acc_sum = 0
        self.train_acc_count = 0

        self.test_acc_sum = 0
        self.test_acc_count = 0

    def forward(self, x):
        predictions = self.c3d_lstm(x)
        return predictions

    @staticmethod
    def compute_batch_accuracy(predictions, labels):
        """
        计算单个 batch 的准确率：至少有一个预测正确即为正确。
        """
        matches = ((predictions > 0.5).float() * labels).sum(dim=1) > 0  # (B,)
        return matches.float().mean().item()  # 返回单个 batch 的平均准确率

    def training_step(self, batch, batch_idx):
        videos = batch['video_frame']  # videos: (B, C, T, H, W), labels: (B, T)
        one_hot_label = batch['one_hot_label']
        predictions = self(videos)  # (B, num_classes)
        loss = self.loss_fn(predictions, one_hot_label[:,:-1])
        # 计算准确率
        acc = self.train_acc(torch.sigmoid(predictions), one_hot_label[:,:-1].int())
        single_acc = self.compute_batch_accuracy(torch.sigmoid(predictions), one_hot_label[:,:-1].int())
        # class_acc = self.train_class_acc(torch.sigmoid(predictions[:,-1].unsqueeze(1)), one_hot_label[:,-1].unsqueeze(1).int())

        self.train_acc_sum += single_acc
        self.train_acc_count += 1

        self.log("train_loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, on_step=True, prog_bar=True, sync_dist=True)
        self.log("single_acc", single_acc, on_step=True, prog_bar=True, sync_dist=True)
        # self.log("class_acc", class_acc, on_step=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        videos = batch['video_frame']  # videos: (B, C, T, H, W), labels: (B, T)
        one_hot_label = batch['one_hot_label']
        predictions = self(videos)
        loss = self.loss_fn(predictions, one_hot_label[:,:-1])
        self.log("val_loss", loss, prog_bar=True)
        # 计算准确率
        acc = self.test_acc(torch.sigmoid(predictions), one_hot_label[:,:-1].int())
        single_acc = self.compute_batch_accuracy(torch.sigmoid(predictions), one_hot_label[:,:-1].int())
        # class_acc = self.test_class_acc(torch.sigmoid(predictions[:,-1].unsqueeze(1) ), one_hot_label[:,-1].unsqueeze(1) .int())

        self.test_acc_sum += single_acc
        self.test_acc_count += 1

        self.log("test_acc", acc, on_step=True, prog_bar=True, sync_dist=True)
        self.log("single_acc", single_acc, on_step=True, prog_bar=True, sync_dist=True)
        # self.log("class_acc", class_acc, on_step=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        """
        在每个 epoch 结束时计算并记录平均准确率。
        """
        avg_acc = self.train_acc_sum / self.train_acc_count
        self.log("train_acc_epoch", avg_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        print("train_acc_epoch:%.3f"%avg_acc)

        # 重置累计变量
        self.train_acc_sum = 0.0
        self.train_acc_count = 0

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器

        :return: 优化器
        """
        optimizer = optim.Adam(self.c3d_lstm.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "test_acc_epoch"}

    def on_validation_epoch_end(self):
        """
        在每个 epoch 结束时计算并记录平均准确率。
        """
        avg_acc = self.test_acc_sum / self.test_acc_count
        self.log("test_acc_epoch", avg_acc, on_epoch=True, prog_bar=True)
        print("test_acc_epoch:%.3f" % avg_acc)

        # 重置累计变量
        self.test_acc_sum = 0.0
        self.test_acc_count = 0

