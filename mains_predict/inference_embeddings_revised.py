from pathlib import Path
import os
import sys
import torch
import argparse

import monai
import numpy as np
import SimpleITK as sitk

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name == "mains_predict" else THIS_FILE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models_vit  # type: ignore # noqa: E402
import models_mae  # type: ignore # noqa: E402


def resample_sitk_image_bspline(sitk_image: sitk.Image, new_size: tuple = (256, 256, 256)) -> sitk.Image:
    """BSpline resampling, matching the SimpleITK-based preprocessing used
    elsewhere in this project (dataset_three_d_fine_resample.py /
    extract_encoder_embeddings.py), rather than MONAI's torch-interpolate-based
    Resize transform (which has no BSpline mode).
    """
    original_size = sitk_image.GetSize()
    original_spacing = sitk_image.GetSpacing()
    new_spacing = [osz * osp / nsz for osz, osp, nsz in zip(original_size, original_spacing, new_size)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(sitk_image)


def load_and_resample_volume(input_path: str, target_size: tuple = (256, 256, 256)) -> torch.Tensor:
    """Replaces monai.transforms.LoadImage + Orientation(RAS) + Resize(bilinear)
    with SimpleITK read + BSpline resample, matching the rest of the pipeline.
    No orientation reordering is applied here, matching Custom3DDataset /
    VolumePathDataset, which also skip explicit reorientation.
    """
    sitk_image = sitk.ReadImage(input_path)
    sitk_image = resample_sitk_image_bspline(sitk_image, new_size=target_size)

    # SimpleITK returns array as (Depth, Height, Width)
    volume = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
    volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  # add channel dim -> [1, D, H, W]
    return volume_tensor


def get_transform_for_inference(input_density=True):
    """CT Scan Preprocessing (see original docstring for full pretraining
    preprocessing description).

    MODIFIED: LoadImage/Orientation(RAS)/Resize(bilinear) have been replaced by
    load_and_resample_volume(), which performs SimpleITK BSpline resampling and
    skips orientation reordering, to match the SimpleITK-based pipeline used
    elsewhere in this project. Only the intensity scaling step remains as a
    MONAI transform below, since its formula is already equivalent to the
    manual HU-clip-and-normalize logic used in that pipeline:
        (v + 1200) / 2000, clipped to [-1200, 800] first.
    """
    preprocess_list = []

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
        wname = "tangerine-checkpoint.pth"
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
        wname = "tangerine-checkpoint.pth"

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
        input_tensor = load_and_resample_volume(input_path, target_size=(256, 256, 256))
        preprocess = get_transform_for_inference(input_density=input_density)
        input_tensor = preprocess(input_tensor)
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
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device >= 0
        else "cpu"
    )
    device = torch.device(device)

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