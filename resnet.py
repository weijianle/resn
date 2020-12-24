import torch
from    torch import  nn
from torch.nn import functional as F
 
# 定义两个卷积层 + 一个短接层
class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(ResBlk, self).__init__()
 
        # 两个卷积层
        self.conv1=nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1=nn.BatchNorm2d(ch_out)
        self.conv2=nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(ch_out)
 
        # 短接层
        self.extra=nn.Sequential()
        if ch_out != ch_in:
            self.extra=nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
                nn.BatchNorm2d(ch_out)
            )
    def forward(self,x):
        """
        :param x: [b,ch,h,w]
        :return:
        """
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
 
        # 短接层
        # element-wise add: [b,ch_in,h,w]=>[b,ch_out,h,w]
        out=self.extra(x)+out
        return out
 
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
 
        # 定义预处理层
        self.conv1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=3,padding=0),
            nn.BatchNorm2d(64)
        )
 
        # 定义堆叠ResBlock层
        # followed 4 blocks
        # [b,64,h,w]-->[b,128,h,w]
        self.blk1=ResBlk(64,128,stride=2)
        # [b,128,h,w]-->[b,256,h,w]
        self.blk2=ResBlk(128,256,stride=2)
        # [b,256,h,w]-->[b,512,h,w]
        self.blk3=ResBlk(256,512,stride=2)
        # [b,512,h,w]-->[b,512,h,w]
        self.blk4=ResBlk(512,512,stride=2)
 
        # 定义全连接层
        self.outlayer=nn.Linear(512,10)
 
    def forward(self,x):
        """
        :param x:
        :return:
        """
        # 1.预处理层
        x=F.relu(self.conv1(x))
 
        # 2. 堆叠ResBlock层：channel会慢慢的增加，  长和宽会慢慢的减少
        # [b,64,h,w]-->[b,512,h,w]
        x=self.blk1(x)
        x=self.blk2(x)
        x=self.blk3(x)
        x=self.blk4(x)
 
        # print("after conv:",x.shape) # [b,512,2,2]
        # 不管原先什么后面两个维度是多少，都化成[1,1]，
        # [b,512,1,1]
        x=F.adaptive_avg_pool2d(x,[1,1])
        # print("after pool2d:",x.shape) # [b,512,1,1]
 
        # 将[b,512,1,1]打平成[b,512*1*1]
        x=x.view(x.size(0),-1)
 
        # 3.放到全连接层，进行打平
        # [b,512]-->[b,10]
        x=self.outlayer(x)
 
        return x
def main():
    blk=ResBlk(64,128,stride=2)
    temp=torch.randn(2,64,32,32)
    out=blk(temp)
    # print('block:',out.shape)
 
    x=torch.randn(2,3,32,32)
    model=ResNet()
    out=model(x)
    # print("resnet:",out.shape)
 
if __name__ == '__main__':
    main()