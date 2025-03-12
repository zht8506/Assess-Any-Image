from gamma_arch import Gamma_MoAE
import torchvision.transforms as tf
from PIL import Image
import torch


if __name__ == "__main__":
    model  = Gamma_MoAE(
        backbone = "../pretrained_weight/UniQA_weight.pt", # you also can use clip weight, just for init model
        pretrained = False,
        num_experts = 3,
        num_ft_layer = 6
    )
    model.training = False

    # load gamma weight
    gamma_weight = "../pretrained_weight/gamma_weight.pth"
    load_net = torch.load(gamma_weight)
    model.load_state_dict(load_net["params"], strict=True)


    img_path = "coffee.jpg" # from koniq10k/512x384/2704811.jpg, gt_label = 0.474
    img_pil = Image.open(img_path).convert('RGB')

    tf_totensor = tf.ToTensor()
    img_tensor = tf_totensor(img_pil)
    img_tensor = img_tensor.unsqueeze(0)

    score = model.forward(img_tensor)
    print(score.detach()) # 0.444