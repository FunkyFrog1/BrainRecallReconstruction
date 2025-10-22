import warnings
from typing import Optional, Union, Literal

import torch
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms


def calculate_pixcorr(original_batch: torch.Tensor,
                      recon_batch: torch.Tensor,
                      eps: float = 1e-8,
                      reduction: str = 'mean') -> torch.Tensor:
    """
    Calculate pixel-wise correlation coefficient (PixCorr) between two batches of images.

    This function computes the Pearson correlation coefficient at pixel level between
    original and reconstructed images, with support for batch processing and multiple
    reduction strategies.

    Args:
        original_batch (torch.Tensor): Batch of original images with shape (B, C, H, W)
        recon_batch (torch.Tensor): Batch of reconstructed images with shape (B, C, H, W)
        eps (float, optional): Small value for numerical stability to prevent division by zero.
                              Defaults to 1e-8.
        reduction (str, optional): Reduction method applied to the output.
                                  Options: 'mean', 'sum', 'none'. Defaults to 'mean'.

    Returns:
        torch.Tensor: Pixel correlation values. Shape depends on reduction:
                     - 'mean': scalar tensor
                     - 'sum': scalar tensor
                     - 'none': tensor of shape (B,)

    Raises:
        ValueError: If input batch shapes don't match or invalid reduction method is provided.

    Example:
        >>> original = torch.randn(4, 3, 256, 256)
        >>> reconstructed = torch.randn(4, 3, 256, 256)
        >>> pixcorr = calculate_pixcorr(original, reconstructed)
        >>> print(f'Pixel correlation: {pixcorr.item():.4f}')
    """
    # Preprocessing: resize images to 512x512
    preprocess = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    original_batch = preprocess(original_batch)
    recon_batch = preprocess(recon_batch)

    # Validate input shapes
    if original_batch.shape != recon_batch.shape:
        raise ValueError(
            f"Input batch shapes must match. Got {original_batch.shape} vs {recon_batch.shape}"
        )

    batch_size, channels, height, width = original_batch.shape

    # Flatten images to (B, C, H*W)
    original_flat = original_batch.view(batch_size, channels, -1)  # (B, C, H*W)
    recon_flat = recon_batch.view(batch_size, channels, -1)  # (B, C, H*W)

    # Compute means along spatial dimension (B, C, 1)
    mean_original = original_flat.mean(dim=2, keepdim=True)
    mean_recon = recon_flat.mean(dim=2, keepdim=True)

    # Center the data
    centered_original = original_flat - mean_original
    centered_recon = recon_flat - mean_recon

    # Compute covariance and variances (B, C)
    covariance = (centered_original * centered_recon).sum(dim=2)
    variance_original = (centered_original ** 2).sum(dim=2)
    variance_recon = (centered_recon ** 2).sum(dim=2)

    # Calculate per-channel Pearson correlation coefficient (B, C)
    correlation_per_channel = covariance / (
            torch.sqrt(variance_original * variance_recon) + eps
    )

    # Average across channels (B,)
    correlation_per_image = correlation_per_channel.mean(dim=1)

    # Apply reduction
    if reduction == 'mean':
        return correlation_per_image.mean()
    elif reduction == 'sum':
        return correlation_per_image.sum()
    elif reduction == 'none':
        return correlation_per_image
    else:
        raise ValueError(
            f"Unsupported reduction method: '{reduction}'. "
            f"Expected one of: 'mean', 'sum', 'none'"
        )


def calculate_ssim(
        original_batch: torch.Tensor,
        recon_batch: torch.Tensor,
        data_range: float = 1.0,
        resize_size: int = 425,
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        use_tqdm: bool = False,
        channel_axis: Optional[int] = -1
) -> Union[torch.Tensor, float]:
    """
    Calculate Structural Similarity Index (SSIM) between original and reconstructed image batches.

    Computes the SSIM metric for image quality assessment, which measures the perceived quality
    by comparing structural information between original and reconstructed images. Supports
    batch processing and multiple reduction strategies.

    Args:
        original_batch (torch.Tensor): Batch of original images with shape (B, C, H, W).
                                       Expected value range [0, 1] or [0, data_range].
        recon_batch (torch.Tensor): Batch of reconstructed images with shape (B, C, H, W).
                                    Same value range as original_batch.
        data_range (float, optional): The data range of the input images (max - min value).
                                      For float images in [0,1], use 1.0. For uint8 in [0,255], use 255.0.
                                      Defaults to 1.0.
        resize_size (int, optional): Target size for resizing images. Defaults to 425.
        reduction (str, optional): Reduction method for batch scores. Options: 'mean', 'sum', 'none'.
                                   Defaults to 'mean'.
        use_tqdm (bool, optional): Whether to display progress bar for batch processing.
                                   Defaults to False.
        channel_axis (int, optional): Axis containing channel information. Set to None for grayscale.
                                      Defaults to -1 for RGB images.

    Returns:
        Union[torch.Tensor, float]: SSIM scores. Shape depends on reduction:
                                   - 'mean': scalar tensor
                                   - 'sum': scalar tensor
                                   - 'none': tensor of shape (B,)

    Raises:
        ValueError: If input batch shapes don't match or invalid parameters are provided.
        RuntimeError: If SSIM computation fails.

    Example:
        >>> original = torch.rand(4, 3, 256, 256)  # Batch of 4 RGB images
        >>> reconstructed = torch.rand(4, 3, 256, 256)
        >>> ssim_score = calculate_ssim(original, reconstructed)
        >>> print(f'SSIM: {ssim_score:.4f}')

        >>> # Get per-image SSIM scores
        >>> ssim_scores = calculate_ssim(original, reconstructed, reduction='none')
        >>> print(f'Per-image SSIM: {ssim_scores}')
    """
    if original_batch.dim() != 4:
        raise ValueError(f"Expected 4D input (B, C, H, W), got {original_batch.dim()}D")

    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")

    # Preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    try:
        original_batch = preprocess(original_batch).float()
        recon_batch = preprocess(recon_batch).float()
    except Exception as e:
        raise RuntimeError(f"Image preprocessing failed: {e}")

    # Convert to NumPy and adjust dimensions (B, H, W, C)
    orig_np = original_batch.permute(0, 2, 3, 1).cpu().numpy()
    recon_np = recon_batch.permute(0, 2, 3, 1).cpu().numpy()

    # Convert to grayscale (B, H, W)
    try:
        orig_gray = rgb2gray(orig_np)
        recon_gray = rgb2gray(recon_np)
    except Exception as e:
        raise RuntimeError(f"Grayscale conversion failed: {e}")

    # Initialize progress bar if requested
    iterator = range(orig_gray.shape[0])
    if use_tqdm:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Calculating SSIM")
        except ImportError:
            warnings.warn("tqdm not available, continuing without progress bar")
            use_tqdm = False

    # Calculate SSIM for each image pair
    ssim_scores = []
    for i in iterator:
        try:
            score = ssim(
                orig_gray[i],
                recon_gray[i],
                data_range=data_range,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                channel_axis=None  # Grayscale images have no channel dimension
            )
            ssim_scores.append(score)
        except Exception as e:
            warnings.warn(f"SSIM computation failed for image {i}: {e}")
            ssim_scores.append(0.0)  # Fallback value

    # Convert to tensor
    ssim_tensor = torch.tensor(ssim_scores, dtype=torch.float32, device=original_batch.device)

    # Apply reduction
    if reduction == 'mean':
        return ssim_tensor.mean()
    elif reduction == 'sum':
        return ssim_tensor.sum()
    elif reduction == 'none':
        return ssim_tensor
    else:
        raise ValueError(
            f"Unsupported reduction method: '{reduction}'. "
            f"Expected one of: 'mean', 'sum', 'none'"
        )
