from plnData import plnDataset
from plnLoss import  plnLoss
from new_resnet import inceptionresnetv2, pretrained_inception
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch


device = 'cuda'
file_root = 'VOCdevkit/VOC2007/JPEGImages/'
batch_size = 1
learning_rate = 0.001
num_epochs = 2

# 自定义训练数据集
train_dataset = plnDataset(img_root=file_root, list_file='voctrain.txt', train=True, transform=[transforms.ToTensor()])
# 加载自定义的训练数据集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True)
# 自定义测试数据集
test_dataset = plnDataset(img_root=file_root, list_file='voctest.txt', train=False, transform=[transforms.ToTensor()])
# 加载自定义的测试数据集
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True)
print('the dataset has %d images' % (len(train_dataset)))


"""
下面这段代码主要适用于迁移学习训练，可以将预训练的ResNet-50模型的参数赋值给新的网络，以加快训练速度和提高准确性。
"""

# net = inceptionresnetv2(num_classes=20, pretrained='imagenet').cuda()
net = pretrained_inception()
# # 是否加载之前训练过的模型
# net_static_dict = torch.load("pln.pth")
# net.load_state_dict(net_static_dict)

criterion = plnLoss(14,2,w_coord=5,w_link=1,w_class=1).to(device)
net.train()

# 定义优化器  RMS
optimizer = torch.optim.RMSprop(
    net.parameters(),
    # 学习率
    lr=learning_rate,
    # 动量
    momentum=0.9,
    # 正则化
    weight_decay=5e-4
)
# Windows环境下使用多进程时需要调用的函数，我训练的时候没用。在Windows下使用多进程需要先将Python脚本打包成exe文件，而freeze_support()的作用就是冻结可执行文件的代码，确保在Windows下正常运行多进程。
# torch.multiprocessing.freeze_support()  # 多进程相关 猜测是使用多显卡训练需要
"""
    这里解释下自己定义参数列表和直接使用net.parameter()的区别：
    在大多数情况下，直接使用net.parameters()和将模型参数放到字典中是没有区别的，
    因为net.parameters()本身就是一个包含模型所有参数的列表。
    但是，如果我们想要对不同的参数设置不同的超参数，那么将模型参数放到字典中会更加方便。
    使用net.parameters()的话，我们需要手动区分不同的参数，
    再分别进行超参数的设置。而将模型参数放到字典中后，我们可以直接对每个参数设置对应的超参数，更加简洁和灵活。
    举个例子，如果我们想要对卷积层和全连接层设置不同的学习率，使用net.parameters()的话，
    我们需要手动区分哪些参数属于卷积层，哪些参数属于全连接层，
    然后分别对这两部分参数设置不同的学习率。而将模型参数放到字典中后，
    我们可以直接对卷积层和全连接层的参数分别设置不同的学习率，更加方便和清晰。
"""
# 开始训练
for epoch in range(num_epochs):
    # 这个地方形成习惯，因为网络可能会用到Dropout和batchnorm
    net.train()
    # 调整学习率
    if epoch == 60:
        learning_rate = 0.0001
    if epoch == 80:
        learning_rate = 0.00001
    # optimizer.param_groups 返回一个包含优化器参数分组信息的列表，每个分组是一个字典，主要包含以下键值：
    # params：当前参数分组中需要更新的参数列表，如网络的权重，偏置等。
    # lr：当前参数分组的学习率。就是我们要提取更新的
    # momentum：当前参数分组的动量参数。
    # weight_decay：当前参数分组的权重衰减参数。
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate      # 更改全部的学习率
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))

    # 计算损失
    total_loss = 0.
    # 开始迭代训练
    for i, (images, target) in enumerate(train_loader):
        # print("training",images.shape,target.shape)
        images, target = images.cuda(), target.cuda()
        pred = net(images)
        # 创建损失函数
        # pred = pred.permute(0, 2, 3, 1)
        loss = criterion(pred, target)
        total_loss += loss.item()

        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()
        if (i + 1) % 5 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (epoch +1, num_epochs, i + 1, len(train_loader), loss.item(), total_loss / (i + 1)))
    # 开始测试
    validation_loss = 0.0
    net.eval()
    for i, (images, target) in enumerate(test_loader):
        images, target = images.cuda(), target.cuda()
        # 输入图像
        pred = net(images)
        # pred = pred.permute(0, 2, 3, 1)
        # 计算损失
        loss = criterion(pred, target)
        # 累加损失
        validation_loss += loss.item()
        # 计算平均loss
    validation_loss /= len(test_loader)

    best_test_loss = validation_loss
    print('get best test loss %.5f' % best_test_loss)
    # 保存模型参数
    torch.save(net.state_dict(), 'pln.pth')


