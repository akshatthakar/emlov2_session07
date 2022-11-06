
## Download images for 10 classes in MNIST
##! wget -O cat.jpeg "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSKr5wT7rfkjkGvNeqgXjBmarC5ZNoZs-H2uMpML8O7Q4F9W-IlUQibBT6IPqyvX45NOgw&usqp=CAU"
##! wget -O Kuvasz.jpeg   "https://upload.wikimedia.org/wikipedia/commons/7/76/Kuvasz_named_Kan.jpg"
##! wget -O English_Springer.jpeg   "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/English_Springer_Spaniel_in_Tallinn.JPG/1280px-English_Springer_Spaniel_in_Tallinn.JPG"
##! wget -O Irish_Setter.jpeg     "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Can_Setter_dog_GFDL.jpg/1024px-Can_Setter_dog_GFDL.jpg"
##! wget -O lady_bug.jpeg  "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Coccinella_magnifica01.jpg/1280px-Coccinella_magnifica01.jpg"
##! wget -O cricket.jpeg  "https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Gryllus_campestris_MHNT.jpg/800px-Gryllus_campestris_MHNT.jpg"
##! wget -O jaguar.jpeg  "https://upload.wikimedia.org/wikipedia/commons/0/0a/Standing_jaguar.jpg"
##! wget -O Grasshopper.jpeg  "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/American_Bird_Grasshopper.jpg/1280px-American_Bird_Grasshopper.jpg"
##! wget -O zebra.jpeg    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Plains_Zebra_Equus_quagga.jpg/800px-Plains_Zebra_Equus_quagga.jpg"
##! wget -O gazelle.jpeg  "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fc/Slender-horned_gazelle_%28Cincinnati_Zoo%29.jpg/1024px-Slender-horned_gazelle_%28Cincinnati_Zoo%29.jpg"
##! wget -O otter.jpeg  "https://upload.wikimedia.org/wikipedia/commons/d/d3/Fischotter%2C_Lutra_Lutra.JPG"


## execute script with below arguments
##explain.py Kuvasz.jpeg  
##explain.py English_Springer.jpeg 
##explain.py Irish_Setter.jpeg  
##explain.py lady_bug.jpeg 
##explain.py cricket.jpeg
##explain.py jaguar.jpeg  
##explain.py Grasshopper.jpeg 
##explain.py zebra.jpeg    
##explain.py gazelle.jpeg  
##explain.py otter.jpeg  

import urllib
import sys

# Download human-readable labels for ImageNet.
# get the classnames

def download_categories():
    url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt","imagenet_classes.txt",)
    urllib.request.urlretrieve(url, filename)
    categories = []
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories

import timm
import torch
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr import Saliency
from captum.attr import DeepLift


def init():
    device = torch.device("cpu")
    model = timm.create_model("resnet18", pretrained=True)
    model.eval()
    model = model.to(device)
    return model

if __name__ == "__main__":
    categories = download_categories()
    model = init()
    image_in = sys.argv[1]
    img_prefix = image_in.split(".")[0] + "_"
    transform = T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor()])
    transform_normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    img = Image.open(image_in)
    transformed_img = transform(img)
    img_tensor = transform_normalize(transformed_img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)
    output = model(img_tensor)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()
    predicted_label = categories[pred_label_idx.item()]
    print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')
    ## integrated_gradient
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(img_tensor, target=pred_label_idx, n_steps=200)
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                    [(0, '#ffffff'),
                                                    (0.25, '#000000'),
                                                    (1, '#000000')], N=256)
    (plt,axis) = viz.visualize_image_attr_multiple(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        cmap=default_cmap,
                                        show_colorbar=True)
    plt.savefig(img_prefix+'integrated_gradient.png')
    ## integrated_gradient_noise_tunnel
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(img_tensor, target=pred_label_idx, n_steps=200)
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                    [(0, '#ffffff'),
                                                    (0.25, '#000000'),
                                                    (1, '#000000')], N=256)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    attributions_ig_nt = noise_tunnel.attribute(img_tensor, nt_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx)
    (plt,axis) = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        cmap=default_cmap,
                                        show_colorbar=True)
    plt.savefig(img_prefix+'integrated_gradient_noise_tunnel.png')
    ## gradient_shap
    default_cmap = LinearSegmentedColormap.from_list('custom blue', [(0, '#ffffff'),(0.25, '#000000'),(1, '#000000')], N=256)
    torch.manual_seed(0)
    np.random.seed(0)
    gradient_shap = GradientShap(model)
    # Defining baseline distribution of images
    rand_img_dist = torch.cat([img_tensor * 0, img_tensor * 1])
    attributions_gs = gradient_shap.attribute(img_tensor,
                                            n_samples=50,
                                            stdevs=0.0001,
                                            baselines=rand_img_dist,
                                            target=pred_label_idx)
    (plt,axis) = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "absolute_value"],
                                        cmap=default_cmap,
                                        show_colorbar=True)
    plt.savefig(img_prefix+'gradient_shap.png')
    ##occlusion
    occlusion = Occlusion(model)
    attributions_occ = occlusion.attribute(img_tensor,
                                        strides = (3, 8, 8),
                                        target=pred_label_idx,
                                        sliding_window_shapes=(3,15, 15),
                                        baselines=0)
    (plt,axis) = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        show_colorbar=True,
                                        outlier_perc=2,)
    plt.savefig(img_prefix+'occlusion.png')
    ## Saliency
    saliency = Saliency(model)
    grads = saliency.attribute(img_tensor, target=285)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    img_tensor.requires_grad = True
    original_image = np.transpose((img_tensor.squeeze(0).cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
    (plt,axis) = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
                            show_colorbar=True, title="Overlayed Gradient Magnitudes")
    plt.savefig(img_prefix+'Saliency.png')

    from captum.robust import FGSM
    from captum.robust import PGD
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(image_in)
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.requires_grad = True
    img_tensor = img_tensor.to(device)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    inv_transform= T.Compose([
        T.Normalize(
            mean = (-1 * np.array(mean) / np.array(std)).tolist(),
            std = (1 / np.array(std)).tolist()
        ),
    ])
    ## GradCAM
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = [ClassifierOutputTarget(281)]
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    rgb_img = inv_transform(img_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    import matplotlib.pyplot as plt
    imgplot = plt.imshow(visualization)
    imgplot.write_png(img_prefix+'GradCAM.png')
    ## GradCAMPlusPlus
    from pytorch_grad_cam import GradCAMPlusPlus
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    rgb_img = inv_transform(img_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    import matplotlib.pyplot as plt
    imgplot = plt.imshow(visualization)
    imgplot.write_png(img_prefix+'GradCAMPlusPlus.png')
