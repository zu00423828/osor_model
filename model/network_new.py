import torch
from torch import nn
from torch.nn import functional as F 
from torch.nn import ModuleList
# import torchsummary
from test_model.transformer import Transformer
from torchvision import models
from test_model.img_utils_torch import batch_mls
from torch.autograd import Variable
from test_old.fom.dense_motion import DenseMotionNetwork
class WarpLoss(nn.Module):
    def __init__(self):
        super(WarpLoss,self).__init__()
    def forward(self,W,M,phat,qhat):
        loss=0
        # b,ctrls,gcol,grow=W.shape
        b,ctrls,_,_,grow, gcol=W.shape
        # phat = phat.reshape(b,ctrls, 2, 1, grow, gcol)                                        # [ctrls, 2, 1, grow, gcol]
        # qhat=qhat.reshape(b,ctrls,2,1,grow,gcol)
        # reshaped_w = W.reshape(b,ctrls, 1, 1, grow, gcol)  
        for i in range(b):
            m=torch.zeros(2,2,grow,gcol,requires_grad=True).cuda()
            for j in range(ctrls):
                m=(M[i][j]*phat[i][j]-qhat[i][j]).pow(2)*W[i][j]
            # print(m.isinf())
            loss+=torch.sum(m)
        return  loss/b
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks =nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1,3,1,1))
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss
class Embedder(nn.Module):
    def __init__(self):
        super(Embedder,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(3,8,3,1,1),nn.BatchNorm2d(8),nn.AvgPool2d(2))
        self.conv2=nn.Sequential(nn.Conv2d(8,8,3,1,1),nn.BatchNorm2d(8),nn.AvgPool2d(2))
        self.conv3=nn.Sequential(nn.Conv2d(8,16,3,1,1),nn.BatchNorm2d(16),nn.AvgPool2d(2))
        self.conv4=nn.Sequential(nn.Conv2d(16,16,3,1,1),nn.BatchNorm2d(16))

        # self.conv2=nn.Sequential(nn.Conv2d(8,128,3,1,1),nn.BatchNorm2d(128),nn.AvgPool2d(2))
        # self.conv3=nn.Sequential(nn.Conv2d(128,256,3,1,1),nn.BatchNorm2d(256),nn.AvgPool2d(2))
        # self.conv4=nn.Sequential(nn.Conv2d(256,256,3,2,1),nn.BatchNorm2d(256),nn.AvgPool2d(2))
        # self.conv5=nn.Sequential(nn.Conv2d(256,256,3,2,1),nn.BatchNorm2d(256),nn.AvgPool2d(2))
        # self.outlayer=nn.Sequential(nn.Conv1d(1,16,3,1,1),nn.BatchNorm1d(16))
    def forward(self,x):
        b,_,_,_=x.shape
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        # x=self.conv5(x)
        x=x.reshape(b,16,1024)
        # x=self.outlayer(x)
        return x

class Patch_Conv(nn.Module):
    def __init__(self,in_channels,out_channels,strides=2,firist_layer=False,last_layer=False):
        super(Patch_Conv,self).__init__()
        if firist_layer:
            self.conv=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=strides,padding=1),nn.LeakyReLU(0.2,True))
        elif last_layer:
            self.conv=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=strides,padding=1),nn.Sigmoid())
        else:
            self.conv=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=strides,padding=1,bias=False),nn.BatchNorm2d(out_channels),nn.LeakyReLU(0.2,True))
    def forward(self,x):
        out=self.conv(x)
        return out


class PatchGan(nn.Module):
    def  __init__(self,input_channel):
        super(PatchGan,self).__init__()
        self.layer1=Patch_Conv(input_channel,64,firist_layer=True)
        self.layer2=Patch_Conv(64,128)
        self.layer3=Patch_Conv(128,256)
        self.layer4=Patch_Conv(256,512,1)
        self.layer5=Patch_Conv(512,1,1,last_layer=True)
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.layer5(out)
        return out

class DoubleConv_Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv_Down,self).__init__()
        self.layer=DoubleConv(in_channels,out_channels)
        self.down=nn.Sequential(nn.Conv2d(out_channels,out_channels,3,2,1),nn.BatchNorm2d(out_channels),nn.ReLU(True))
    def forward(self,x):
        out=self.layer(x)
        return out,self.down(out)
class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.layer=nn.Sequential(nn.Conv2d(in_channels,out_channels,3,1,1),nn.BatchNorm2d(out_channels),nn.ReLU(True),
                            nn.Conv2d(out_channels,out_channels,3,1,1),nn.BatchNorm2d(out_channels),nn.ReLU(True))
    def forward(self,x):
        return self.layer(x)
class Upblock(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels):
        super(Upblock,self).__init__()
        self.up=nn.Sequential(nn.ConvTranspose2d(mid_channels,out_channels,3,2,1,1))
        self.layer=DoubleConv(in_channels,mid_channels)
    def forward(self,x,skip):
        x=torch.cat([x,skip],dim=1)
        x=self.layer(x)
        x=self.up(x)
        return x
class Unet(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Unet,self).__init__()
        encoder=[]
        layer_specs=[
        64,
        64*2,
        64*4,
        64*8,
        ]
        for idx,out_channels in enumerate(layer_specs):
            if idx==0:
                layer=DoubleConv_Down(in_channel,out_channels)
            elif idx==len(layer_specs)-1:
                layer=DoubleConv(layer_specs[idx-1],out_channels)
            else:
                layer=DoubleConv_Down(layer_specs[idx-1],out_channels,)
            encoder.append(layer)
        self.encoder=nn.ModuleList(encoder)
        self.dencoder1=Upblock(529,512,256)
        self.dencoder2=Upblock(512,256,128)
        self.dencoder3=Upblock(256,128,64)
        self.dencoder4=DoubleConv(128,64)
        self.last=nn.Sequential(nn.Conv2d(64,3,1),nn.Tanh())
        # layer_specs=[
        #     64*4,
        #     64*2,
        #     64
        # ]
        # layer=[]
        # for idx,out_channels in enumerate(layer_specs):
        #     if idx==0:
        #         pass
    def forward(self,x,warp):
        feature_list=[]
        for f in self.encoder[:-1]:
            feature,x=f(x)
            feature_list.append(feature)
        x=self.encoder[-1](x)
        x=self.dencoder1(x,warp["mask"])
        x=self.dencoder2(x,feature_list[-1])
        x=self.dencoder3(x,feature_list[-2])
        x=torch.cat([x,feature_list[-3]],dim=1)
        x=self.dencoder4(x)
        x=self.last(x)
        return x



            

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.embedder=Embedder()
        self.transformer=Transformer(emb_dims=16,n_blocks=1,dropout=0,ff_dims=1024,n_heads=4)
        # self.motion_estimaton=MotionEstimation()
        self.warpgenerator=DenseMotionNetwork(32,5,1024,16,3,scale_factor=0.125)
        # self.generate_moudle=Pix2Pix(3,3)
        self.generate_moudle=Unet(3,3)
    def forward(self,x1,x2):
        em_s=self.embedder(x1)
        em_d=self.embedder(x2)
        kp_s,kp_d=self.transformer(em_s,em_d)
        # return kp_s,kp_d
        # me,M,W,phat,qhat=self.motion_estimaton(kp_s,kp_d)
        me,W,M,phat,qhat=batch_mls(kp_s,kp_d)
        warpping=self.warpgenerator(x1,kp_d,kp_d)
        out=self.generate_moudle(x1,warpping)
        
        return out,warpping,W,M,phat,qhat#,warpping_img
