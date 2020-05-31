from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

path_img = '../../day3-DataLoader/data/rmb_split/train/1/0B89KOA3.jpg'
img = Image.open(path_img)
plt.imshow(img)
plt.show()

# resize
Resize = transforms.Resize((32, 32))
resize_img = Resize(img)
plt.imshow(resize_img)
plt.show()

# 随机裁剪
RandomCrop = transforms.RandomCrop(32, padding=4)
random_image = RandomCrop(resize_img)
plt.imshow(random_image)
plt.show()