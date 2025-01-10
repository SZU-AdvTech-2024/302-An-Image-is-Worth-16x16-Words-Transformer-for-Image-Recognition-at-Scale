import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchinfo import summary
import data_setup, engine
from helper_functions import download_data, set_seeds, plot_loss_curves
import model, DATASET
import matplotlib.pyplot as plt
import vit


class CustomViT(torchvision.models.VisionTransformer):
    def __init__(self, pretrained_weights=None):
        super().__init__(image_size=224,
                         patch_size=16,
                         num_layers=12,
                         num_heads=12,
                         hidden_dim=768,
                         mlp_dim=3072)
        if pretrained_weights is not None:
            self.load_state_dict(pretrained_weights)
        self.heads_weight = nn.Parameter(torch.randn((1, 196), requires_grad=True))
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor (如果需要)
        x = self._process_input(x)  # 确保这个方法是可用的

        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        # Classifier "token" as used by standard language architectures
        cls = x[:, 0]
        x = x[:, 1:]
        x = torch.matmul(self.softmax(self.heads_weight), x)
        x = torch.squeeze(x, dim=1)
        x = self.heads(x+cls)

        return x

if __name__ == '__main__':
    # Get automatic transforms from pretrained ViT weights

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Get pretrained weights for ViT-Base
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

    # 2. Setup a ViT model instance with pretrained weights
    pretrained_vit = CustomViT(
        pretrained_weights=torchvision.models.vit_b_16(weights=pretrained_vit_weights).state_dict()).to(device)

    # 3. Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    # 4. Change the classifier head (set the seeds to ensure same initialization with linear head)
    set_seeds()
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=3).to(device)
    # pretrained_vit # uncomment for model output
    # # Print a summary using torchinfo (uncomment for actual output)
    summary(model=pretrained_vit,
            input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )


    image_path = download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
        destination="pizza_steak_sushi")
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    pretrained_vit_transforms = pretrained_vit_weights.transforms()
    # Setup dataloaders
    train_dataloader_pretrained, test_dataloader_pretrained, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=pretrained_vit_transforms,
        batch_size=32)  # Could increase if we had more samples, such as here: https://arxiv.org/abs/2205.01580 (there are other improvements there too...)

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(params=pretrained_vit.parameters(),
                                 lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the classifier head of the pretrained ViT feature extractor model
    set_seeds()
    pretrained_vit_results = engine.train(model=pretrained_vit,
                                          train_dataloader=train_dataloader_pretrained,
                                          test_dataloader=test_dataloader_pretrained,
                                          optimizer=optimizer,
                                          loss_fn=loss_fn,
                                          epochs=10,
                                          device=device)
    # Plot the loss curves
    from helper_functions import plot_loss_curves

    plot_loss_curves(pretrained_vit_results)
    plt.show()
    # Save the model
    # import utils
    # utils.save_model(model=pretrained_vit,
    #                  target_dir="models",
    #                  model_name="08_pretrained_vit_feature_extractor_pizza_steak_sushi.pth")

