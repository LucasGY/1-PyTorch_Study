from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

path_img = '../../day3_DataLoader/data/rmb_split/train/1/0B89KOA3.jpg'
img = Image.open(path_img).convert('RGB')
plt.imshow(img)
plt.show()

# resize
Resize = transforms.Resize((500, 500))  # 原图像size>>resize:导致图片过于模糊
resize_img = Resize(img)
plt.imshow(resize_img)
plt.show()

# 随机裁剪
RandomCrop = transforms.RandomCrop(500, padding=4)
random_image = RandomCrop(resize_img)
plt.imshow(random_image)
plt.show()