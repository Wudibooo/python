# author:Wudibooo
# time:2021/4/19:16:52

#1  加载库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms


#2  定义超参数
BATCH_SIZE = 64   #每次处理的数据
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10 #训练的轮次


#3  构建pipeline，对图像做处理
pipeline = transforms.Compose([
    transforms.ToTensor(), #将图片转换成tensor
    transforms.Normalize((0.1307,), (0.3081,)) #正则化，降低模型复杂度
])


#4  下载，加载数据 mnist
from torch.utils.data import DataLoader
#下载
train_set = datasets.MNIST('Mnistdata', train=True, download=True, transform=pipeline)  #transform通道
test_set = datasets.MNIST('Mnistdata', train=False, download=True, transform=pipeline)
#加载
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  #shuffle打乱图片
test_loader =DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


# #将一个图片可视
# import cv2          #opencv
# import numpy as np
#
# with open('./Mnistdata/MNIST/raw/train-images-idx3-ubyte','rb') as f:
#     file = f.read()
# image1 = [int(str(item).encode('ascii'),16) for item in file [16:16+784]] #存储格式要求从16开始
#
# image1_np = np.array(image1, dtype=np.uint8).reshape(28,28,1)
# print(image1_np.shape)
#
# cv2.imwrite('test1.jpg',image1_np)


#5  构建网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,5)  #输入通道1 输出通道10 kenel—size = 5 padding = 0
        self.conv2 = nn.Conv2d(10,20,3)  #输入通道10 输出20  kenel-size = 3 padding = 0
        self.fc1 = nn.Linear(20*10*10,500) #输入通道20*10*10,输出500  全连接
        self.fc2 = nn.Linear(500,10)  #500：输入通道  10#输出通道      全连接
    
    def forward(self,x):
        input_size = x.size(0) #x的第一个维度的大小，就是一批的图片数量的多少

        x = self.conv1(x)   #输入 batch * 1 * 28*28 输出 batch*10*24*24
        x = F.relu(x)  #激活函数 y=x x>0, y=0 x<0       10*24*24
        x = F.max_pool2d(x, 2, 2)  #最大池化层  stride = 2  输出 shape10*12*12
        x = self.conv2(x)   #卷积操作   input batch*10*12*12   output batch*20*10*10
        x = F.relu(x)  #激活函数

        x = x.view(input_size,-1)#将x按照一个二维tensor打平，-1 自动计算维度

        x = self.fc1(x)   #全连接成 batch*2000 -》 batch*500
        x = F.relu(x)       #激活函数

        x = self.fc2(x)  #全链接层

        output = F.log_softmax(x, dim=1)  #计算分类组
        return output

#6  定义优化器
model = Digit().to(DEVICE)

optimizer = optim.Adam(model.parameters())  #Adam优化

#7  训练
def train_model(model, device, train_loader, optimizer, epoch):
    #模型训练
    model.train()

    for batch_index, (data, target) in enumerate(train_loader):
                #enumerate(支持迭代对象, [start=0(下标起始位置)])
                # 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        #部署到Device上去
        data, target = data.to(device), target.to(device)
        #梯度初始化为0
        optimizer.zero_grad()
        #训练后的结果
        output = model(data)
        #计算损失   分类问题交叉熵比较好用
        loss = F.cross_entropy(output,target) #output预测值 ，target真实值
        #找到概率值最大的下标
        pred = output.max(1, keepdim = True) #pred = output.argmax(dim=1)
        #反向传播
        loss.backward()
        #参数优化
        optimizer.step()
        #打印数据
        if batch_index % 3000 == 0 :
            print('Train Epoch:{} \t Loss :{:0.6f}'.format(epoch,loss.item()))


#8  测试
def test_model(model, device, test_loader):
    #模型认证
    model.eval()  ##eval()时，模型会自动把BN和DropOut固定住，不会取平均，而是用训练好的值
    #正确率的计算
    correct = 0.0
    #测试损失
    test_loss = 0.0
    with torch.no_grad(): #不会计算梯度，也不会进行反向传播
        for data,target in test_loader:
            #部署到device
            data, target = data.to(device), target.to(device)
            #测试数据
            output = model(data)
            #计算损失
            test_loss += F.cross_entropy(output, target).item()
            #找到概率值最大的下标
            pred = output.max(1, keepdim = True)[1] #值，索引
            #pred = torch.max(output,dim= 1)
            #pred = output.argmax(dim=1)
            #累计正确的值
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('test -- Average loss : {:.4f}, Accuracy : {:.3f}\n'.format(test_loss, 100.0 * correct / len(test_loader.dataset)))

#9  调用方法
# for epoch in range(1,EPOCHS+1):
#     train_model(model, DEVICE, train_loader, optimizer, epoch)
#     test_model(model, DEVICE, test_loader)

# 保存训练的model
# torch.save({'state_dict': model.state_dict()}, './训练好的模型/CNN_Tommy_weight2.pth.tar')

#测试自己的图片
from PIL import Image
import numpy as np
#  加载参数
ckpt = torch.load('./训练好的模型/CNN_Tommy_weight2.pth.tar')
model.load_state_dict(ckpt['state_dict'])            #参数加载到指定模型cnn

#检测自己的数字集 [9,2,4,5,7,8,7,1,3,4,8,2,6,5,1,3,9,6]
print('9,2,4,5,7,8,7,1,3,4,8,2,6,5,1,3,9,6')
for i in range(1,19):
    #  要识别的图片
    input_image = './自己手写数字/{}.jpg'.format(i)

    im = Image.open(input_image).resize((28, 28))     #取图片数据
    im = im.convert('L')      #灰度图

    im_data = np.array(im)
    im_data = torch.from_numpy(im_data).float()

    im_data = im_data.view(1, 1, 28, 28).to(DEVICE)  #转为tensor shape 1*1*28*28
    out = model(im_data)
    pred = out.max(1, keepdim = True)[1]     #输出最大值的标签

    print(pred.item(),end=',')

