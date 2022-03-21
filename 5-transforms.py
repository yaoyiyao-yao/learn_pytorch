from PIL import Image
from tensorboardX import SummaryWriter
from torchvision import transforms

img_path = "dataset/train/bees_image/39747887_42df2855ee.jpg"
img_PIL = Image.open(img_path)

writer = SummaryWriter("logs")

#transforms.toTensor()
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img_PIL)
writer.add_image("bee",img_tensor,1)

#transforms.Normalize
trans_norm = transforms.Normalize([1,2,3],[7,7,7])
img_norm = trans_norm(img_tensor)
writer.add_image("bee",img_norm,2)

#transforms.Resize
trans_resize = transforms.Resize((100,100))
img_size = trans_resize(img_PIL)
writer.add_image("bee",tensor_trans(img_size),3)

#transform.Compse
trans_compose = transforms.Compose([trans_resize,tensor_trans])
img_compse = trans_compose(img_PIL)
writer.add_image("bee",img_compse,4)

#transform.RandomCrop
trans_random  = transforms.RandomCrop(100)
trans_compose_2 = transforms.Compose([trans_random,tensor_trans])
for i in range(10):
    img_crop = trans_compose_2(img_PIL)
    writer.add_image("RandomCrop",img_crop,i)

writer.close()
