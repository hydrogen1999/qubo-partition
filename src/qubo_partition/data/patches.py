from PIL import Image
import numpy as np


def split_image_into_patches(image: Image.Image, patch_size=512, stride=None):
    """
    Split an image into non-overlapping patches.

    Returns:
        patches: list of PIL Images
        coords: list of (x, y) coordinates
        original_size: (width, height)
    """

    width, height = image.size
    if stride is None:
       stride = patch_size

    patches = []
    coords = []

    for y in range(0, height, stride):
        for x in range(0, width, stride):

            right = min(x + patch_size, width)
            bottom = min(y + patch_size, height)

            patch = image.crop((x, y, right, bottom))

            patches.append(patch)
            coords.append((x, y))

    return patches, coords, (width, height)


def stitch_masks(masks, coords, original_size):
    width, height = original_size

    votes = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    for mask, (x, y) in zip(masks, coords):
        h, w = mask.shape

        votes[y:y+h, x:x+w] += mask.astype(np.float32)
        counts[y:y+h, x:x+w] += 1

    counts[counts == 0] = 1

    averaged = votes / counts

    return averaged >= 0.5

def split_mask_into_patches(mask: np.ndarray, patch_size=512, stride=None):
    """
    Split a binary mask into patches.

    Returns:
        patches
        coords
        original_size
    """

    height, width = mask.shape
    if stride is None:
       stride = patch_size

    patches = []
    coords = []

    for y in range(0, height, stride):
        for x in range(0, width, stride):

            bottom = min(y + patch_size, height)
            right = min(x + patch_size, width)

            patch = mask[y:bottom, x:right]

            patches.append(patch)
            coords.append((x, y))

    return patches, coords, (width, height)

def split_seed_into_patches(seed_mask: np.ndarray, patch_size=512):
    """
    Split a foreground/background seed mask into non-overlapping patches.

    Returns:
        patches: list of numpy arrays
        coords: list of (x, y)
        original_size: (width, height)
    """

    height, width = seed_mask.shape

    patches = []
    coords = []

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):

            bottom = min(y + patch_size, height)
            right = min(x + patch_size, width)

            patch = seed_mask[y:bottom, x:right]

            patches.append(patch)
            coords.append((x, y))

    return patches, coords, (width, height)