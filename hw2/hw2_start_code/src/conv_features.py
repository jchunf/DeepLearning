#Leverage a properneural network visualization toolkitto visualize someconvfeatures [8pts].
import torch.nn
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms
from PIL import Image
from visualize_fuctions import conv_viz
plt.rcParams.update({'figure.max_open_warning': 0})
count = 0

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('~/hw2/src/best_model_adam_a.pt', map_location=device)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    trans = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for name, m in model.named_modules():
        # show conv features
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_pre_hook(conv_viz)

    img = Image.open('~/hw2/hw2_dataset/test/AnnualCrop/AnnualCrop_1111.jpg')
    img = trans(img).unsqueeze(0).cuda() if torch.cuda.is_available() else trans(img).unsqueeze(0)
    with torch.no_grad():
        model(img)