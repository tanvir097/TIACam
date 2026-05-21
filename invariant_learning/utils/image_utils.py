import torchvision.transforms.functional as TF


def tensor_batch_to_pil_list(tensor_batch):
    pil_images = []

    for img in tensor_batch:
        img = img.detach().cpu()
        pil_images.append(TF.to_pil_image(img))

    return pil_images