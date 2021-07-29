import numpy as np
import torch
import PIL.Image as pillow


def img2tensor(
    img: pillow.Image,
    size: tuple = None,
) -> torch.Tensor:
    """
    Args:
        img: image to convert
        size: (W, H) of output tensor
    Returns:
        tensor of shape (1, 3, H, W)
            the first dimension (batch size) is neccesary for the CNN
            3 channels are for RGB (alpha channel is not supported)
            values are between 0 and 1
    Examples:
        >>> img = pillow.new('RGB', (16, 9), color=(0, 100, 255))
        >>> tensor = img2tensor(img)
        >>> tensor.shape
        torch.Size([1, 3, 9, 16])
        >>> torch.all((0 <= tensor) & (tensor <= 1))
        tensor(True)
    """
    if size is not None:
        img = img.resize(size, pillow.BILINEAR)
    arr = np.array(img).transpose(2, 0, 1)[np.newaxis, ...]
    return torch.tensor(arr).float() / 255


def tensor2img(tensor: torch.Tensor) -> pillow.Image:
    """
    Args:
        tensor: shape (1, 3, H, W),
                values are between 0 and 1
    Returns:
        pillow image of size (W, H)
    Examples:
        >>> x = torch.ones((1, 3, 9, 16))
        >>> torch.all((0 <= x) & (x <= 1))
        tensor(True)
        >>> tensor2img(x).size
        (16, 9)
    """
    tensor = tensor.squeeze(0) * 255
    arr = np.uint8(tensor.numpy()).transpose(1, 2, 0)
    return pillow.fromarray(arr)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
