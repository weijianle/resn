import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn,optim
from visdom import Visdom
# from lenet5 import  Lenet5
from resnet import ResNet18
import time
 
def main():
    batch_siz=32
    cifar_train = datasets.CIFAR10('cifar',True,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),download=True)
    cifar_train=DataLoader(cifar_train,batch_size=batch_siz,shuffle=True)
 
    cifar_test = datasets.CIFAR10('cifar',False,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),download=True)
    cifar_test=DataLoader(cifar_test,batch_size=batch_siz,shuffle=True)
 
    x,label = iter(cifar_train).next()
    print('x:',x.shape,'label:',label.shape)
 
    # 指定运行到cpu //GPU
    device=torch.device('cpu')
    # model = Lenet5().to(device)
    model = ResNet18().to(device)
 
    # 调用损失函数use Cross Entropy loss交叉熵
    # 分类问题使用CrossEntropyLoss比MSELoss更合适
    criteon = nn.CrossEntropyLoss().to(device)
    # 定义一个优化器
    optimizer=optim.Adam(model.parameters(),lr=1e-3)
    print(model)
 
    viz=Visdom()
    viz.line([0.],[0.],win="loss",opts=dict(title='Lenet5 Loss'))
    viz.line([0.],[0.],win="acc",opts=dict(title='Lenet5 Acc'))
 
    # 训练train
    for epoch in range(1000):
        # 变成train模式
        model.train()
        # barchidx:下标，x:[b,3,32,32],label:[b]
        str_time=time.time()
        for barchidx,(x,label) in enumerate(cifar_train):
            # 将x，label放在gpu上
            x,label=x.to(device),label.to(device)
            # logits:[b,10]
            # label:[b]
            logits = model(x)
            loss = criteon(logits,label)
 
            # viz.line([loss.item()],[barchidx],win='loss',update='append')
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(barchidx)
        end_time=time.time()
        print('第 {} 次训练用时: {}'.format(epoch,(end_time-str_time)))
        viz.line([loss.item()],[epoch],win='loss',update='append')
        print(epoch,'loss:',loss.item())
 
 
        # 变成测试模式
        model.eval()
        with torch.no_grad():
            #  测试test
            # 正确的数目
            total_correct=0
            total_num=0
            for x,label in cifar_test:
                # 将x，label放在gpu上
                x,label=x.to(device),label.to(device)
                # [b,10]
                logits=model(x)
                # [b]
                pred=logits.argmax(dim=1)
                # [b] = [b'] 统计相等个数
                total_correct+=pred.eq(label).float().sum().item()
                total_num+=x.size(0)
            acc=total_correct/total_num
            print(epoch,'acc:',acc)
            print("------------------------------")
 
            viz.line([acc],[epoch],win='acc',update='append')
            # viz.images(x.view(-1, 3, 32, 32), win='x')
 
 
if __name__ == '__main__':
    main()