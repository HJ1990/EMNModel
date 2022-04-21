import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ExperienceAttention(nn.Module):
    def __init__(self, batch_size):
        super(ExperienceAttention, self).__init__()
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
        proj_query = self.proj_query.view(m_batchsize, -1, width * height).permute(0, 2, 1).float()  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height).float()  # B X C x (*W*H)
        energy = torch.bmm(proj_key, proj_query)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(attention.permute(0, 2, 1), proj_value)
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out




class EMN(nn.Module):

    def __init__(self):
        super(EMN, self).__init__()

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

        self.attention_module_1 = MemoryModule(19, './memory_features_1.npy')
        self.attention_module_2 = MemoryModule(19, './memory_features_2.npy')
        self.attention_module_3 = MemoryModule(19, './memory_features_3.npy')
        self.attention_module_4 = MemoryModule(19, './memory_features_4.npy')
        self.attention_module_5 = MemoryModule(19, './memory_features_5.npy')

        self.feature_fuse_5 = nn.Conv2d(512, 19, 1, padding=(0, 0))  # features fusion
        self.feature_restore_5 = nn.Conv2d(19, 64, 1, padding=(0, 0)) # features restoration
        
        self.feature_fuse_4 = nn.Conv2d(512, 19, 1, padding=(0, 0))  # features fusion
        self.feature_restore_4 = nn.Conv2d(19, 64, 1, padding=(0, 0))  # features restoration

        self.feature_fuse_3 = nn.Conv2d(256, 19, 1, padding=(0, 0))  # features fusion
        self.feature_restore_3 = nn.Conv2d(19, 64, 1, padding=(0, 0))  # features restoration
  

        self.feature_fuse_2 = nn.Conv2d(128, 19, 1, padding=(0, 0))  # features fusion
        self.feature_restore_2 = nn.Conv2d(19, 64, 1, padding=(0, 0))  # features restoration


        self.feature_fuse_1 = nn.Conv2d(64, 19, 1, padding=(0, 0))  # features fusion
        self.feature_restore_1 = nn.Conv2d(19, 64, 1, padding=(0, 0))  # features restoration


        self.fuse_conv1 = nn.Conv2d(64*5, 256, 3, padding=(1, 1))  # features fusion
        self.fuse_conv2 = nn.Conv2d(256, 256, 1, padding=(0, 0))  # features fusion
        self.score = nn.Conv2d(256, 19, 1, padding=(0, 0))

        self._initialize_weights()

    def forward(self, x):
        # x.size(0)即为batch_size
        H = x.size(2)
        W = x.size(3)

        out = self.conv1_1(x)  # 222
        out = F.relu(out)
        out = self.conv1_2(out)  # 222
        out = F.relu(out)
        en_f1 = self.maxpool1(out)  # 112

        out = self.conv2_1(en_f1)  # 110
        out = F.relu(out)
        out = self.conv2_2(out)  # 110
        out = F.relu(out)
        en_f2 = self.maxpool2(out)  # 56

        out = self.conv3_1(en_f2)  # 54
        out = F.relu(out)
        out = self.conv3_2(out)  # 54
        out = F.relu(out)
        out = self.conv3_3(out)  # 54
        out = F.relu(out)
        en_f3 = self.maxpool3(out)  # 28

        out = self.conv4_1(en_f3)  # 26
        out = F.relu(out)
        out = self.conv4_2(out)  # 26
        out = F.relu(out)
        out = self.conv4_3(out)  # 26
        out = F.relu(out)
        en_f4 = self.maxpool4(out)  # 14

        out = self.conv5_1(en_f4)  # 12
        out = F.relu(out)
        out = self.conv5_2(out)  # 12
        out = F.relu(out)
        out = self.conv5_3(out)  # 12
        out = F.relu(out)
        out = self.maxpool5(out)  # 7
        en_f5 = F.relu(out)

        fuse_features_5 = F.relu(self.feature_fuse_5(en_f5))
        attention_features_5 = self.attention_module_5(fuse_features_5)
        attention_features_5 = self.feature_restore_5(attention_features_5)


        fuse_features_4 = F.relu(self.feature_fuse_4(en_f4))
        attention_features_4 = self.attention_module_4(fuse_features_4)
        attention_features_4 = self.feature_restore_5(attention_features_4)

        fuse_features_3 = F.relu(self.feature_fuse_3(en_f3))
        attention_features_3 = self.attention_module_3(fuse_features_3)
        attention_features_3 = self.feature_restore_3(attention_features_3)


        fuse_features_2 = F.relu(self.feature_fuse_2(en_f2))
        attention_features_2 = self.attention_module_2(fuse_features_2)
        attention_features_2 = self.feature_restore_2(attention_features_2)


        fuse_features_1 = F.relu(self.feature_fuse_1(en_f1))
        attention_features_1 = self.attention_module_1(fuse_features_1)
        attention_features_1 = self.feature_restore_1(attention_features_1)


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
        fuse_features = F.relu(self.fuse_conv1(fusecat))
        fuse_features = F.relu(self.fuse_conv2(fuse_features))
        fuse_features = F.relu(self.score(fuse_features))
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