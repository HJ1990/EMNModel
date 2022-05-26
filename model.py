import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models


class LocationAttention(nn.Module):
    def __init__(self, batch_size):
        super(LocationAttention, self).__init__()
        self.batch_size = batch_size

    def forward(self, x1, x2):
        feature_batch = x1.shape[0]
        feature_channel = x1.shape[1]
        feature_dimensions = x1.shape[2]*x1.shape[3]
        x1_linear = x1.view(feature_batch, feature_channel, feature_dimensions)
        x2_linear = x2.view(feature_batch, feature_channel, feature_dimensions)
        x1_linear = x1_linear.permute(0, 2, 1)
        f = torch.matmul(x1_linear.float(), x2_linear.float())
        f = F.tanh(f)
        f = torch.matmul(f, x1_linear)
        return f


class MemoryModule(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, memory_path):
        super(MemoryModule, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        query = np.load(memory_path)
        self.proj_query = torch.from_numpy(query).unsqueeze(0).cuda()
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.proj_query.repeat(m_batchsize, 1, 1, 1).view(m_batchsize, -1, width * height).permute(0, 2, 1).float()  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height).float()  # B X C x (*W*H)
        energy = torch.bmm(proj_key, proj_query)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(attention.permute(0, 2, 1), proj_value)
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()

        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=(1, 1))  # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2))  # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 1, padding=(0, 0))  # 512 * 30 * 30
        # self.fuse = nn.Conv2d(512, 64, 3, padding=(1, 1))  # 64 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 7 * 7

        self.fc = nn.Sequential(
            nn.Linear(512 * 20 * 13, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 14),
        )

        self._initialize_weights()

    def forward(self, x):
        # x.size(0)即为batch_size
        in_size = x.size(0)

        out = self.conv1_1(x)  # 222
        out = F.relu(out)
        out = self.conv1_2(out)  # 222
        out = F.relu(out)
        out = self.maxpool1(out)  # 112

        out = self.conv2_1(out)  # 110
        out = F.relu(out)
        out = self.conv2_2(out)  # 110
        out = F.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 54
        out = F.relu(out)
        out = self.conv3_2(out)  # 54
        out = F.relu(out)
        out = self.conv3_3(out)  # 54
        out = F.relu(out)
        out = self.maxpool3(out)  # 28

        out = self.conv4_1(out)  # 26
        out = F.relu(out)
        out = self.conv4_2(out)  # 26
        out = F.relu(out)
        out = self.conv4_3(out)  # 26
        out = F.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 12
        out = F.relu(out)
        out = self.conv5_2(out)  # 12
        out = F.relu(out)
        out = self.conv5_3(out)  # 12
        out = F.relu(out)
        out = self.maxpool5(out)  # 7
        out = F.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)



        return out

    def _initialize_weights(self):
        # print(self.modules())

        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                # print(m.weight.data.type())
                # input()
                # m.weight.data.fill_(1.0)
                nn.init.xavier_uniform_(m.weight, gain=1)


class MemoryFeatures(nn.Module):

    def __init__(self):
        super(MemoryFeatures, self).__init__()

        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=(1, 1))  # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2))  # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 1, padding=(0, 0))  # 512 * 30 * 30
        # self.fuse = nn.Conv2d(512, 64, 3, padding=(1, 1))  # 64 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 7 * 7

        self.fc = nn.Sequential(
            nn.Linear(512 * 20 * 13, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 14),
        )
        self._initialize_weights()

    def forward(self, x):
        # x.size(0)即为batch_size
        in_size = x.size(0)

        out = self.conv1_1(x)  # 222
        out = F.relu(out)
        out = self.conv1_2(out)  # 222
        out = F.relu(out)
        out = self.maxpool1(out)  # 112
        out_features_1 = out

        out = self.conv2_1(out)  # 110
        out = F.relu(out)
        out = self.conv2_2(out)  # 110
        out = F.relu(out)
        out = self.maxpool2(out)  # 56
        out_features_2 = out

        out = self.conv3_1(out)  # 54
        out = F.relu(out)
        out = self.conv3_2(out)  # 54
        out = F.relu(out)
        out = self.conv3_3(out)  # 54
        out = F.relu(out)
        out = self.maxpool3(out)  # 28
        out_features_3 = out

        out = self.conv4_1(out)  # 26
        out = F.relu(out)
        out = self.conv4_2(out)  # 26
        out = F.relu(out)
        out = self.conv4_3(out)  # 26
        out = F.relu(out)
        out = self.maxpool4(out)  # 14
        out_features_4 = out

        out = self.conv5_1(out)  # 12
        out = F.relu(out)
        out = self.conv5_2(out)  # 12
        out = F.relu(out)
        out = self.conv5_3(out)  # 12
        out = F.relu(out)
        out = self.maxpool5(out)  # 7
        out_features_5 = out
        out = F.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        feature_sum_1 = out_features_1[:, 0, :, :] + out_features_1[:, 1, :, :]
        for i in range(2, 64):
            feature_sum_1 = feature_sum_1 + out_features_1[:, i, :, :]

        feature_sum_2 = out_features_2[:, 0, :, :] + out_features_2[:, 1, :, :]
        for i in range(2, 128):
            feature_sum_2 = feature_sum_2 + out_features_2[:, i, :, :]

        feature_sum_3 = out_features_3[:, 0, :, :] + out_features_3[:, 1, :, :]
        for i in range(2, 256):
            feature_sum_3 = feature_sum_3 + out_features_3[:, i, :, :]

        feature_sum_4 = out_features_4[:, 0, :, :] + out_features_4[:, 1, :, :]
        for i in range(2, 512):
            feature_sum_4 = feature_sum_4 + out_features_4[:, i, :, :]

        feature_sum_5 = out_features_5[:, 0, :, :]+out_features_5[:, 1, :, :]
        for i in range(2, 256):
            feature_sum_5 = feature_sum_5 + out_features_5[:, i, :, :]
        return out, feature_sum_1, feature_sum_2, feature_sum_3, feature_sum_4, feature_sum_5

    def _initialize_weights(self):
        # print(self.modules())

        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                # print(m.weight.data.type())
                # input()
                # m.weight.data.fill_(1.0)
                nn.init.xavier_uniform_(m.weight, gain=1)


class VGG16WithMemory(nn.Module):

    def __init__(self):
        super(VGG16WithMemory, self).__init__()
        cls_num = 14
        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=(1, 1))  # 64 * 222 * 222
        # self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 64 * 222* 222
        self.bn1_2 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        # self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        self.bn2_2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d((2, 2))  # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        # self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        # self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.bn3_3 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        # self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        # self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.bn4_3 = nn.BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        # self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        # self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 1, padding=(0, 0))  # 512 * 30 * 30
        self.bn5_3 = nn.BatchNorm2d(512)
        # self.fuse = nn.Conv2d(512, 64, 3, padding=(1, 1))  # 64 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 7 * 7

        self.attention_module_1 = MemoryModule(cls_num, 'F:/FashionParsing/PaperDoll/experience_matrix1.npy')
        self.attention_module_2 = MemoryModule(cls_num, 'F:/FashionParsing/PaperDoll/experience_matrix2.npy')
        self.attention_module_3 = MemoryModule(cls_num, 'F:/FashionParsing/PaperDoll/experience_matrix3.npy')
        self.attention_module_4 = MemoryModule(cls_num, 'F:/FashionParsing/PaperDoll/experience_matrix4.npy')
        self.attention_module_5 = MemoryModule(cls_num, 'F:/FashionParsing/PaperDoll/experience_matrix5.npy')

        self.feature_fuse_5 = nn.Conv2d(512, cls_num, 1, padding=(0, 0))  # features fusion
        self.feature_restore_5 = nn.Conv2d(cls_num, 64, 1, padding=(0, 0)) # features restoration
        # self.decode5_1 = nn.Conv2d(512, 64, 3, padding=(1, 1))
        # self.decode5_2 = nn.Conv2d(64, 1, 1, padding=(0, 0))

        self.feature_fuse_4 = nn.Conv2d(512, cls_num, 1, padding=(0, 0))  # features fusion
        self.feature_restore_4 = nn.Conv2d(cls_num, 64, 1, padding=(0, 0))  # features restoration
        # self.decode4_1 = nn.Conv2d(512, 64, 3, padding=(1, 1))
        # self.decode4_2 = nn.Conv2d(64, 1, 1, padding=(0, 0))

        self.feature_fuse_3 = nn.Conv2d(256, cls_num, 1, padding=(0, 0))  # features fusion
        self.feature_restore_3 = nn.Conv2d(cls_num, 64, 1, padding=(0, 0))  # features restoration
        # self.decode3_1 = nn.Conv2d(256, 64, 3, padding=(1, 1))
        # self.decode3_2 = nn.Conv2d(64, 1, 1, padding=(0, 0))

        self.feature_fuse_2 = nn.Conv2d(128, cls_num, 1, padding=(0, 0))  # features fusion
        self.feature_restore_2 = nn.Conv2d(cls_num, 64, 1, padding=(0, 0))  # features restoration
        # self.decode2_1 = nn.Conv2d(128, 64, 3, padding=(1, 1))
        # self.decode2_2 = nn.Conv2d(64, 1, 1, padding=(0, 0))

        self.feature_fuse_1 = nn.Conv2d(64, cls_num, 1, padding=(0, 0))  # features fusion
        self.feature_restore_1 = nn.Conv2d(cls_num, 64, 1, padding=(0, 0))  # features restoration
        # self.decode1_1 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        # self.decode1_2 = nn.Conv2d(64, 1, 1, padding=(0, 0))

        self.fuse_conv1 = nn.Conv2d(64*5, 256, 3, padding=(1, 1))  # features fusion
        self.fuse_conv2 = nn.Conv2d(256, 256, 1, padding=(0, 0))  # features fusion
        self.score = nn.Conv2d(256, 14, 1, padding=(0, 0))

        self._initialize_weights()

    def forward(self, x):
        # x.size(0)即为batch_size
        H = x.size(2)
        W = x.size(3)

        out = (self.conv1_1(x))  # 222
        out = F.leaky_relu(out)
        out = (self.conv1_2(out))  # 222
        out = F.leaky_relu(out)
        en_f1 = self.maxpool1(out)  # 112

        out = (self.conv2_1(en_f1))  # 110
        out = F.leaky_relu(out)
        out = (self.conv2_2(out))  # 110
        out = F.leaky_relu(out)
        en_f2 = self.maxpool2(out)  # 56

        out = (self.conv3_1(en_f2))  # 54
        out = F.leaky_relu(out)
        out = (self.conv3_2(out))  # 54
        out = F.leaky_relu(out)
        out = (self.conv3_3(out))  # 54
        out = F.leaky_relu(out)
        en_f3 = self.maxpool3(out)  # 28

        out = (self.conv4_1(en_f3))  # 26
        out = F.leaky_relu(out)
        out = (self.conv4_2(out))  # 26
        out = F.leaky_relu(out)
        out = (self.conv4_3(out))  # 26
        out = F.leaky_relu(out)
        en_f4 = self.maxpool4(out)  # 14

        out = (self.conv5_1(en_f4) ) # 12
        out = F.leaky_relu(out)
        out = (self.conv5_2(out))  # 12
        out = F.leaky_relu(out)
        out = (self.conv5_3(out))  # 12
        out = F.leaky_relu(out)
        out = self.maxpool5(out)  # 7
        en_f5 = F.leaky_relu(out)

        fuse_features_5 = F.leaky_relu(self.feature_fuse_5(en_f5))
        attention_features_5 = self.attention_module_5(fuse_features_5)
        attention_features_5 = self.feature_restore_5(attention_features_5)
        # de_f5 = F.relu(self.decode5_1(attention_features_5))
        # de_f5 = F.relu(self.decode5_2(de_f5))

        fuse_features_4 = F.leaky_relu(self.feature_fuse_4(en_f4))
        attention_features_4 = self.attention_module_4(fuse_features_4)
        attention_features_4 = self.feature_restore_5(attention_features_4)
        # de_f4 = F.relu(self.decode4_1(attention_features_4))
        # de_f4 = F.relu(self.decode4_2(de_f4))

        fuse_features_3 = F.leaky_relu(self.feature_fuse_3(en_f3))
        attention_features_3 = self.attention_module_3(fuse_features_3)
        attention_features_3 = self.feature_restore_3(attention_features_3)
        # de_f3 = F.relu(self.decode3_1(attention_features_3))
        # de_f3 = F.relu(self.decode3_2(de_f3))

        fuse_features_2 = F.leaky_relu(self.feature_fuse_2(en_f2))
        attention_features_2 = self.attention_module_2(fuse_features_2)
        attention_features_2 = self.feature_restore_2(attention_features_2)
        # de_f2 = F.relu(self.decode2_1(attention_features_2))
        # de_f2 = F.relu(self.decode2_2(de_f2))

        fuse_features_1 = F.leaky_relu(self.feature_fuse_1(en_f1))
        attention_features_1 = self.attention_module_1(fuse_features_1)
        attention_features_1 = self.feature_restore_1(attention_features_1)
        # de_f1 = F.relu(self.decode1_1(attention_features_1))
        # de_f1 = F.relu(self.decode1_2(de_f1))

        weight_deconv1 = make_bilinear_weights(4, 64).cuda()
        weight_deconv2 = make_bilinear_weights(8, 64).cuda()
        weight_deconv3 = make_bilinear_weights(16, 64).cuda()
        weight_deconv4 = make_bilinear_weights(32, 64).cuda()
        weight_deconv5 = make_bilinear_weights(64, 64).cuda()

        de_f1 = torch.nn.functional.conv_transpose2d(attention_features_1, weight_deconv1, stride=2)
        de_f2 = torch.nn.functional.conv_transpose2d(attention_features_2, weight_deconv2, stride=4)
        de_f3 = torch.nn.functional.conv_transpose2d(attention_features_3, weight_deconv3, stride=8)
        de_f4 = torch.nn.functional.conv_transpose2d(attention_features_4, weight_deconv4, stride=16)
        de_f5 = torch.nn.functional.conv_transpose2d(attention_features_5, weight_deconv5, stride=32)

        de_f1 = crop(de_f1, H, W)
        de_f2 = crop(de_f2, H, W)
        de_f3 = crop(de_f3, H, W)
        de_f4 = crop(de_f4, H, W)
        de_f5 = crop(de_f5, H, W)

        fusecat = torch.cat((de_f1, de_f2, de_f3, de_f4, de_f5), dim=1)
        fuse_features = F.leaky_relu(self.fuse_conv1(fusecat))
        fuse_features = F.leaky_relu(self.fuse_conv2(fuse_features))
        fuse_features = F.leaky_relu(self.score(fuse_features))
        # res = torch.nn.LogSoftmax(dim=1)
        # res_label = torch.autograd.Variable(torch.max(res, dim=1)[1].float(), requires_grad=True)
        return fuse_features

    def _initialize_weights(self):
        # print(self.modules())

        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                # print(m.weight.data.type())
                # input()
                # m.weight.data.fill_(1.0)
                nn.init.xavier_uniform_(m.weight, gain=1)


def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


def crop(variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[:, :, y1: y1 + th, x1: x1 + tw]


class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.down(x)
        return out


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=1000, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Sequential(
            nn.Linear(2048 * 7 * 13, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 14),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet101():
    return ResNet([3, 4, 23, 3])


class ResNetFeature(nn.Module):
    def __init__(self,blocks, num_classes=14, expansion = 4):
        super(ResNetFeature,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Sequential(
            nn.Linear(2048 * 7 * 13, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 14),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        out_features_1 = x
        x = self.layer1(x)
        out_features_2 = x
        x = self.layer2(x)
        out_features_3 = x
        x = self.layer3(x)
        out_features_4 = x
        x = self.layer4(x)
        out_features_5 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        feature_sum_1 = out_features_1[:, 0, :, :] + out_features_1[:, 1, :, :]
        for i in range(2, 64):
            feature_sum_1 = feature_sum_1 + out_features_1[:, i, :, :]

        feature_sum_2 = out_features_2[:, 0, :, :] + out_features_2[:, 1, :, :]
        for i in range(2, 64):
            feature_sum_2 = feature_sum_2 + out_features_2[:, i, :, :]

        feature_sum_3 = out_features_3[:, 0, :, :] + out_features_3[:, 1, :, :]
        for i in range(2, 128):
            feature_sum_3 = feature_sum_3 + out_features_3[:, i, :, :]

        feature_sum_4 = out_features_4[:, 0, :, :] + out_features_4[:, 1, :, :]
        for i in range(2, 256):
            feature_sum_4 = feature_sum_4 + out_features_4[:, i, :, :]

        feature_sum_5 = out_features_5[:, 0, :, :] + out_features_5[:, 1, :, :]
        for i in range(2, 512):
            feature_sum_5 = feature_sum_5 + out_features_5[:, i, :, :]
        return out, feature_sum_1, feature_sum_2, feature_sum_3, feature_sum_4, feature_sum_5



class ResnetWithMemory(nn.Module):
    def __init__(self, blocks, expansion=4):
        super(ResnetWithMemory, self).__init__()
        cls_num = 14
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=3, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.attention_module_1 = MemoryModule(cls_num, 'F:/FashionParsing/PaperDoll/resnet_experience_matrix1.npy')
        self.attention_module_2 = MemoryModule(cls_num, 'F:/FashionParsing/PaperDoll/resnet_experience_matrix2.npy')
        self.attention_module_3 = MemoryModule(cls_num, 'F:/FashionParsing/PaperDoll/resnet_experience_matrix3.npy')
        self.attention_module_4 = MemoryModule(cls_num, 'F:/FashionParsing/PaperDoll/resnet_experience_matrix4.npy')
        self.attention_module_5 = MemoryModule(cls_num, 'F:/FashionParsing/PaperDoll/resnet_experience_matrix5.npy')

        self.feature_fuse_5 = nn.Conv2d(2048, cls_num, 1, padding=(0, 0))  # features fusion
        self.feature_restore_5 = nn.Conv2d(cls_num, 64, 1, padding=(0, 0)) # features restoration

        self.feature_fuse_4 = nn.Conv2d(1024, cls_num, 1, padding=(0, 0))  # features fusion
        self.feature_restore_4 = nn.Conv2d(cls_num, 64, 1, padding=(0, 0))  # features restoration

        self.feature_fuse_3 = nn.Conv2d(512, cls_num, 1, padding=(0, 0))  # features fusion
        self.feature_restore_3 = nn.Conv2d(cls_num, 64, 1, padding=(0, 0))  # features restoration

        self.feature_fuse_2 = nn.Conv2d(256, cls_num, 1, padding=(0, 0))  # features fusion
        self.feature_restore_2 = nn.Conv2d(cls_num, 64, 1, padding=(0, 0))  # features restoration

        self.feature_fuse_1 = nn.Conv2d(64, cls_num, 1, padding=(0, 0))  # features fusion
        self.feature_restore_1 = nn.Conv2d(cls_num, 64, 1, padding=(0, 0))  # features restoration

        self.fuse_conv1 = nn.Conv2d(64*5, 256, 3, padding=(1, 1))  # features fusion
        self.fuse_conv2 = nn.Conv2d(256, 256, 1, padding=(0, 0))  # features fusion
        self.score = nn.Conv2d(256, 14, 1, padding=(0, 0))

        self._initialize_weights()

    def forward(self, x):
        # x.size(0)即为batch_size
        H = x.size(2)
        W = x.size(3)
        en_f1 = self.conv1(x)

        en_f2 = self.layer1(en_f1)
        en_f3 = self.layer2(en_f2)
        en_f4 = self.layer3(en_f3)
        en_f5 = self.layer4(en_f4)

        fuse_features_5 = F.leaky_relu(self.feature_fuse_5(en_f5))
        attention_features_5 = self.attention_module_5(fuse_features_5)
        attention_features_5 = self.feature_restore_5(attention_features_5)

        fuse_features_4 = F.leaky_relu(self.feature_fuse_4(en_f4))
        attention_features_4 = self.attention_module_4(fuse_features_4)
        attention_features_4 = self.feature_restore_5(attention_features_4)

        fuse_features_3 = F.leaky_relu(self.feature_fuse_3(en_f3))
        attention_features_3 = self.attention_module_3(fuse_features_3)
        attention_features_3 = self.feature_restore_3(attention_features_3)

        fuse_features_2 = F.leaky_relu(self.feature_fuse_2(en_f2))
        attention_features_2 = self.attention_module_2(fuse_features_2)
        attention_features_2 = self.feature_restore_2(attention_features_2)

        fuse_features_1 = F.leaky_relu(self.feature_fuse_1(en_f1))
        attention_features_1 = self.attention_module_1(fuse_features_1)
        attention_features_1 = self.feature_restore_1(attention_features_1)

        weight_deconv1 = make_bilinear_weights(4, 64).cuda()
        weight_deconv2 = make_bilinear_weights(8, 64).cuda()
        weight_deconv3 = make_bilinear_weights(16, 64).cuda()
        weight_deconv4 = make_bilinear_weights(32, 64).cuda()
        weight_deconv5 = make_bilinear_weights(64, 64).cuda()

        de_f1 = torch.nn.functional.conv_transpose2d(attention_features_1, weight_deconv2, stride=4)
        de_f2 = torch.nn.functional.conv_transpose2d(attention_features_2, weight_deconv2, stride=4)
        de_f3 = torch.nn.functional.conv_transpose2d(attention_features_3, weight_deconv3, stride=8)
        de_f4 = torch.nn.functional.conv_transpose2d(attention_features_4, weight_deconv4, stride=16)
        de_f5 = torch.nn.functional.conv_transpose2d(attention_features_5, weight_deconv5, stride=32)

        de_f1 = crop(de_f1, H, W)
        de_f2 = crop(de_f2, H, W)
        de_f3 = crop(de_f3, H, W)
        de_f4 = crop(de_f4, H, W)
        de_f5 = crop(de_f5, H, W)

        fusecat = torch.cat((de_f1, de_f2, de_f3, de_f4, de_f5), dim=1)
        fuse_features = F.leaky_relu(self.fuse_conv1(fusecat))
        fuse_features = F.leaky_relu(self.fuse_conv2(fuse_features))
        fuse_features = F.leaky_relu(self.score(fuse_features))
        return fuse_features

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)

    def _initialize_weights(self):

        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
