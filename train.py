import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchinfo import summary
import data_setup, engine
from helper_functions import download_data, set_seeds, plot_loss_curves
import model, DATASET
if __name__ == '__main__':

    train_dataloader, test_dataloader, class_names = DATASET.getdata()
    vit = model.ViT(num_classes=len(class_names))
    optimizer = torch.optim.Adam(params=vit.parameters(),
                                 lr=1e-3,
                                 betas=(0.9, 0.999),
                                 weight_decay=0.3)

    loss = nn.CrossEntropyLoss()

    set_seeds()

    results = engine.train(model=vit,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss,
                           epochs=10,
                           device='cuda')

    from helper_functions import plot_loss_curves

    plot_loss_curves(results)
    plt.show()