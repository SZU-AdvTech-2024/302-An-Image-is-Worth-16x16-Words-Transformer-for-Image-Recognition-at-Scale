import torch
from torch import nn
from torchinfo import summary
from torchvision import transforms
import helper_functions
from pretrain import CustomViT
model_path = './models/08_pretrained_vit_feature_extractor_pizza_steak_sushi.pth'
custom_image_path = './data/pizza_steak_sushi/test/pizza/195160.jpg'
model = CustomViT()
model.heads = nn.Linear(in_features=768, out_features=3)
summary(model=model,
        input_size=(32, 3, 224, 224),  # (batch_size, color_channels, height, width)
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])
trans = transforms.Resize((224, 224))
model.load_state_dict(torch.load(model_path, weights_only=True))

helper_functions.pred_and_plot_image(model=model,
                    transform=trans,
                    image_path=custom_image_path,
                    class_names=['pizza', 'steak', 'sushi'])

