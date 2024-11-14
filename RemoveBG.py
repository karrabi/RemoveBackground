from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
# model.to('cuda')
model.to('cpu')
model.eval()

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('apple.jpg')
plt.imshow(image)
plt.show(block=False)
plt.pause(3)
# input_images = transform_image(image).unsqueeze(0).to('cuda')
input_images = transform_image(image).unsqueeze(0).to('cpu')

# Prediction
with torch.no_grad():
    preds = model(input_images)[-1].sigmoid().cpu()
pred = preds[0].squeeze()
pred_pil = transforms.ToPILImage()(pred)
mask = pred_pil.resize(image.size)
image.putalpha(mask)
plt.clf() 
plt.imshow(image)
plt.show()
image.save("no_bg_image.png")