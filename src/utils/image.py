import numpy as np

def extract_patches(image, patch_size=64, overlap=0.5):
    """
    Extracts overlapping patches from an image.

    Args:
        image (np.ndarray): Input 2D image.
        patch_size (int): Size of each patch (default is 64).
        overlap (float): Fraction of overlap between patches (default is 0.5).

    Returns:
        List[np.ndarray]: List of extracted patches.
        List[Tuple[int, int]]: List of top-left coordinates for each patch.
    """
    stride = int(patch_size * (1 - overlap))
    patches = []
    positions = []

    for i in range(0, image.shape[0] - patch_size + 1, stride):
        for j in range(0, image.shape[1] - patch_size + 1, stride):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
            positions.append((i, j))

    return patches, positions

def reconstruct_image(patches, positions, image_shape):
    """
    Reconstructs an image from patches and their positions.

    Args:
        patches (List[np.ndarray]): List of patches.
        positions (List[Tuple[int, int]]): Corresponding top-left positions of patches.
        image_shape (Tuple[int, int]): Shape of the original image.

    Returns:
        np.ndarray: Reconstructed image.
    """
    patch_size = patches[0].shape[0]
    reconstructed = np.zeros(image_shape, dtype=np.float32)
    weight = np.zeros(image_shape, dtype=np.float32)

    for patch, (i, j) in zip(patches, positions):
        reconstructed[i:i + patch_size, j:j + patch_size] = np.add(reconstructed[i:i + patch_size, j:j + patch_size], patch)
        weight[i:i + patch_size, j:j + patch_size] += 1

    weight[weight == 0] = 1
    reconstructed /= weight
    return reconstructed