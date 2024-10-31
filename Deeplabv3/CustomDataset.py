import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import numpy as np

# 自定义数据集
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None,mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = os.listdir(images_dir)
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")
        image_extension = os.path.splitext(self.images[idx])[1]
        mask_name = os.path.join(self.masks_dir, self.images[idx].replace(image_extension, '.png'))
        mask = Image.open(mask_name).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask)
        return image, mask.squeeze(0)
    

# 数据预处理 和 数据增强 随机旋转、翻转、缩放和颜色抖动
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Nearest to keep class values intact
    transforms.ToTensor(),
])
mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),  # Nearest to keep class values intact
    transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))
])

# 加载数据集
# 划分数据集为训练集和测试集
train_dataset = SegmentationDataset('deeplabv3/train/', 'deeplabv3/masks/train1/', transform=transform, mask_transform=mask_transform)
val_dataset = SegmentationDataset('deeplabv3/val/', 'deeplabv3/masks/val1/', transform=transform, mask_transform=mask_transform)
# test_dataset = SegmentationDataset('deeplabv3/test/', 'deeplabv3/masks/test1/', transform=transform, mask_transform=mask_transform)
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 加载 DeepLabV3 模型
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
num_classes = 4  # 根据您的类别数调整
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  # 替换分类层 
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 GPU 或 CPU



# 训练设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 改进了权重 更加重视代表性不足的类
weights = torch.tensor([0.5, 1.5, 2.5, 3.0], dtype=torch.float32).to(device)  
criterion = nn.CrossEntropyLoss(weight=weights)
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# 训练循环
num_epochs = 5  # 根据需要设置训练轮数
class_pixel_count = np.zeros(num_classes)  # Initialize class pixel count
for epoch in range(num_epochs):
    model.train()  # 训练模型
    for images, masks in train_loader:
        images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
        masks = masks.to('cuda' if torch.cuda.is_available() else 'cpu')
        for cls in range(num_classes):
            class_pixel_count[cls] += (masks == cls).sum().item()  # Count pixels for each class
            # print(f"Batch pixel count for class {cls}: {(masks == cls).sum().item()}")
        # 清空梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)['out']
        
        # 计算损失
        loss = criterion(outputs, masks.long())  # 确保 mask 是 long 类型
        
        # 反向传播
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), f'checkpoint{epoch+1}.pth')
torch.save(model.state_dict(), f'deeplabv3_model.pth')    
print("Training complete. Model saved.")
