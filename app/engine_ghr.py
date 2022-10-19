from app.ghr_utils.model import BiSeNet
import torch
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms 
import gc

# Deteccion de sombrero y lentes
def evaluate_ghr(image):
    """ Recibe una imagen y la procesa para saber si existe lentes o sombrero """

    gc.enable()
    n_classes = 19
    
    net = BiSeNet(n_classes=n_classes)
    save_pth = osp.join('./app/ghr_utils/res/cp', '79999_iter.pth')
    net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():

        imagen = Image.open(image).convert('RGB')
        imagen_resize = imagen.resize((512, 512), Image.Resampling.BILINEAR)
        imagen_tensor = to_tensor(imagen_resize)
        imagen_unsqueeze = torch.unsqueeze(imagen_tensor, 0)

        out = net(imagen_unsqueeze)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        del imagen_resize, imagen_tensor, imagen_unsqueeze

        gc.collect()

        # Regresa una lista con los items encontrados
        return np.unique(parsing).tolist()