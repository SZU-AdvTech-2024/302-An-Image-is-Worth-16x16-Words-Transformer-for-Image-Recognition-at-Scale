# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms
from torchinfo import summary
import data_setup, engine
from helper_functions import download_data, set_seeds, plot_loss_curves
import model

def getdata(BATCH_SIZE = 32):
    image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                               destination="pizza_steak_sushi")

    train_dir = image_path / "train"
    test_dir = image_path / "test"

    IMG_SIZE = 224
    manual_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=manual_transforms, # use manually created transforms
        batch_size=BATCH_SIZE
    )

    # visualization
    # Get a batch of images
    image_batch, label_batch = next(iter(train_dataloader))
    # Get a single image from the batch
    image, label = image_batch[0], label_batch[0]
    # Plot image with matplotlib
    plt.figure()
    plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
    plt.title(class_names[label])
    plt.axis(False)
    # Create example values
    height = 224 # H ("The training resolution is 224.")
    width = 224 # W
    color_channels = 3 # C
    patch_size = 16 # P

    # Calculate N (number of patches)
    number_of_patches = int((height * width) / patch_size**2)
    print(f"Number of patches (N) with image height (H={height}), width (W={width}) and patch size (P={patch_size}): {number_of_patches}")
    # Input shape (this is the size of a single image)
    embedding_layer_input_shape = (height, width, color_channels)

    # Output shape
    embedding_layer_output_shape = (number_of_patches, patch_size**2 * color_channels)

    print(f"Input shape (single 2D image): {embedding_layer_input_shape}")
    print(f"Output shape (single 2D image flattened into patches): {embedding_layer_output_shape}")
    # Setup hyperparameters and make sure img_size and patch_size are compatible
    num_patches = IMG_SIZE / patch_size
    assert IMG_SIZE % patch_size == 0, "Image size must be divisible by patch size"
    print(f"Number of patches per row: {num_patches}\
            \nNumber of patches per column: {num_patches}\
            \nTotal patches: {num_patches * num_patches}\
            \nPatch size: {patch_size} pixels x {patch_size} pixels")
    image_permuted = image.permute(1, 2, 0)
    # Create a series of subplots
    fig, axs = plt.subplots(nrows=IMG_SIZE // patch_size,  # need int not float
                            ncols=IMG_SIZE // patch_size,
                            figsize=(num_patches, num_patches),
                            sharex=True,
                            sharey=True)

    # Loop through height and width of image
    for i, patch_height in enumerate(range(0, IMG_SIZE, patch_size)):  # iterate through height
        for j, patch_width in enumerate(range(0, IMG_SIZE, patch_size)):  # iterate through width

            # Plot the permuted image patch (image_permuted -> (Height, Width, Color Channels))
            axs[i, j].imshow(image_permuted[patch_height:patch_height + patch_size,  # iterate through height
                             patch_width:patch_width + patch_size,  # iterate through width
                             :])  # get all color channels

            # Set up label information, remove the ticks for clarity and set labels to outside
            axs[i, j].set_ylabel(i + 1,
                                 rotation="horizontal",
                                 horizontalalignment="right",
                                 verticalalignment="center")
            axs[i, j].set_xlabel(j + 1)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].label_outer()

    # Set a super title
    # fig.suptitle(f"{class_names[label]} -> Patchified", fontsize=16)
    # # plt.show()
    # # set_seeds()
    #
    # # Create an instance of patch embedding layer
    # patchify = model.PatchEmbedding(in_channels=3,
    #                                 patch_size=16,
    #                                 embedding_dim=768)

    # Pass a single image through
    # print(f"Input image shape: {image.unsqueeze(0).shape}")
    # patch_embedded_image = patchify(
    #     image.unsqueeze(0))  # add an extra batch dimension on the 0th index, otherwise will error
    # print(f"Output patch embedding shape: {patch_embedded_image.shape}")
    plt.show()
    return train_dataloader, test_dataloader, class_names


if __name__ == '__main__':
     train_dataloader, test_dataloader, class_names = getdata()





