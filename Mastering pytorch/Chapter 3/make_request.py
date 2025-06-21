import io
import json
import requests
from PIL import Image

from torchvision import transforms

image = Image.open('./sample2.jpg')


def load_image(image, transform=None):
    img = image.convert('RGB')
    img = img.resize([224, 224], Image.LANCZOS)
    transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), 
                         (0.229, 0.224, 0.225))])
    if transform is not None:
        img = transform(img)
    
    return img

image_tensor = load_image(image)

dimensions = io.StringIO(json.dumps({'dims': list(image_tensor.shape)}))
data = io.BytesIO(bytearray(image_tensor.numpy()))

r = requests.post('http://localhost:8885/image_caption', files = {'metadata': dimensions, 'data': data})

response = r.content.decode('utf-8').strip()

print(response)