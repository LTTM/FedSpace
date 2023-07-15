def normalization(style: str = 'center'):
    '''Return mean and standard deviation for different styles of normalization.
    '''
    if style == 'center':
        return ((0.5,), (0.5,))
    if style == 'imagenet':
        return ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    raise ValueError(f'Unrecognized normalization style: "{style}"')


def to_tensor(normalize: str = 'center'):
    '''Create a torchvision.transform that converts an image to a tensor, and optionally
    normalizes with a mean and standard deviation corresponding to the supplied normalization
    style.
    '''
    from torchvision import transforms
    t = transforms.ToTensor()
    if normalize is not None:
        t = transforms.Compose([t, transforms.Normalize(*normalization(normalize))])
    return t
