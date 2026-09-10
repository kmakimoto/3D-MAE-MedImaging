"""
File: inference_embedding.py
Author: Ariel H. Curiale
Github: https://gitlab.com/Curiale
Date: 2025-11-21
Description: Inference of the Tangerine model to compute image embeddings for
CT
"""

from pathlib import Path
import os
import torch
import argparse

import monai
import numpy as np

# import torch.nn.functional as F

from config.local_paths import TANGERINE_REPO_DIR

os.sys.path.append(TANGERINE_REPO_DIR)

import models_vit  # type: ignore # noqa: E402
import models_mae  # type: ignore # noqa: E402



def get_transform_for_inference(input_density=True):
    """CT Scan Preprocessing:
    Dataset preprocessing and augmentation for model pretraining All volumes were
    resampled to a uniform size of 256×256×256. To preserve each scan’s physical
    dimensions, the new voxel spacing was dynamically computed based on the
    original voxel spacing and image dimensions, using the formula:
    new_spacing =(original_size × original_spacing)/ new_size. Essential spatial
    metadata, including the origin, orientation, and direction, were preserved
    during this process. Following resampling, all processed volumes were
    visually inspected by two board-certified radiologists to exclude corrupted
    files, non-thoracic anatomy, or incomplete thoracic coverage, ensuring a
    high-quality dataset for model training (see Supplementary Data 1).
    Importantly, scans with variable slice thickness or imaging artefacts
    were retained to preserve the diversity and realism of real-world clinical
    imaging data. CT voxel intensities were clipped to a Hounsfield Unit r
    ange of [-1200, 800] focusing on the clinically relevant intensity range,
    and normalised to a range of [0, 1] using min–max scaling. To optimise
    computational efficiency during pretraining, data augmentation was limited
    to random flipping along the sagittal and axial planes, maintaining
    anatomical fidelity while introducing sufficient variability into the dataset.

    Im using RAS to be the same as the Merlin FM, changing RAS to LPS is not
    having a positive impact in the embeedings, they are almost the same

    [1] https://www.nature.com/articles/s43856-025-01328-1
    """
    preprocess_list = [
        monai.transforms.LoadImage(),
        monai.transforms.EnsureType(),
        monai.transforms.EnsureChannelFirst(),
        monai.transforms.Orientation(axcodes="RAS"),
        monai.transforms.Resize(
            spatial_size=(256, 256, 256),
            mode="bilinear",
        ),
    ]

    if input_density:
        preprocess_list.append(
            monai.transforms.ScaleIntensityRange(
                a_min=-1200,  # Min HU value
                a_max=800,  # Max HU value
                b_min=0,  # Target min
                b_max=1,  # Target max
                clip=True,  # Clip values outside range
            )
        )
    else:
        # |J| was cliped to |J| >= 0.01 and then saved as |J| * 1e5
        preprocess_list.append(
            monai.transforms.ScaleIntensityRange(
                a_min=0.01 * 1e5,  # Min |J| value
                a_max=10 * 1e5,  # Max |J| value Clip huge expansions (x10)
                b_min=0,  # Target min
                b_max=1,  # Target max
                clip=True,  # Clip values outside range
            )
        )

    preprocess = monai.transforms.Compose(preprocess_list)
    return preprocess



def get_model(mversion, checkpoints_dir="./checkpoints"):

    # Image input should be 256 x 256 x 256
    if mversion == "mae_vit_large_patch16_dec512d8b":
        # Full MAE model with ViT. Use this model for training using the mask
        # AE. Avoid to use for inference the latent space.
        model = models_mae.__dict__[mversion](norm_pix_loss=False)
        wname = "tangerine_mae_vit_large_patch16_dec512d8b.pth"
    else:
        # Only ViT for embedding inference
        # force to use the ViT inference model
        mversion == "vit_large_patch16_yo"
        model = models_vit.__dict__[mversion](
            num_classes=1,
            drop_path_rate=0,
            global_pool=True,
        )
        # Checkpoint weights corresponds to the MAE model with ViT
        wname = "tangerine_mae_vit_large_patch16_dec512d8b.pth"

    print(f"Loading model {mversion}")

    checkpoint_path = os.path.join("checkpoints", wname)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model = checkpoint["model"]
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    return model, mversion


def inference_step(
    model,
    input_path,
    output_path,
    label,
    label_path,
    device,
    input_density=True,
):
    # Run inference
    if label is None:
        preprocess = get_transform_for_inference(input_density=input_density)
        input_tensor = preprocess(input_path)
        input_tensors = {"image": input_tensor}
    else:
        preprocess = get_transform_for_inference_with_mask()
        input_tensors = preprocess({"image": input_path, "label": label_path})
        if label.lower() == "lung" or label.lower() == "lobes":
            input_tensors = get_lobe_crops(
                input_tensors, input_density=input_density
            )
        else:
            raise ValueError(f"Label {label} not supported")

    # Add the batch dimmension
    input_tensors = {k: v[None] for k, v in input_tensors.items()}

    embeddings = {}
    for k, v in input_tensors.items():
        # Scale and normalize the input
        with torch.inference_mode():
            with torch.autocast(device.type, dtype=torch.bfloat16):
                batch_img = v.to(device)
                if isinstance(model, models_vit.VisionTransformer):
                    # Return the latent space and if global_pool is True then
                    # it retunrs the [cls+mean_patches] = 2*latent
                    outputs = model.forward_features(batch_img)
                elif isinstance(model, models_mae.MaskedAutoencoderViT):
                    # Return the latent, mask, ids_restore
                    mask_ratio = 0.0  # Do not mask
                    outputs, _, _ = model.forward_encoder(
                        batch_img, mask_ratio
                    )
                else:
                    raise ValueError(f"Model {model} not supported")

            outputs = outputs.detach().to(torch.float32).cpu()
            # Do not normalize it will destroy the magnitude
            outputs = outputs.numpy().squeeze()
            # outputs = F.normalize(outputs, p=2, dim=-1).numpy().squeeze()

        reduce_size = False
        if reduce_size:
            # save the embeddings as float16 to reduce size
            outputs = outputs.astype(np.float16)
        embeddings[k] = outputs

    save_path = Path(output_path).parent
    save_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **embeddings)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input CT scan file path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output embedding file path",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        help="Label to use for the embedding",
    )
    parser.add_argument(
        "-lp",
        "--label_path",
        type=str,
        help="Label path file to use for the embedding",
    )
    parser.add_argument("-id", "--input_density", action="store_true")
    parser.set_defaults(input_density=True)
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        default=0,
        help="Device to use for inference",
    )
    parser.add_argument(
        "-n",
        "--model_name",
        type=str,
        default="vit_large_patch16_yo",
        help="Model name to use for inference",
    )
    args = parser.parse_args()

    device = (
        f"cuda:{args.id_device}"
        if torch.cuda.is_available() and args.id_device >= 0
        else "cpu"
    )

    input_path = args.input
    output_path = args.output
    label = args.label
    label_path = args.label_path
    input_density = args.input_density
    mname = args.model_name

    # Create the model
    model, mname = get_model(mname)
    model = model.to(device)
    model.eval()

    inference_step(
        model,
        input_path,
        output_path,
        label,
        label_path,
        device,
        input_density=input_density,
    )
