from transformers import CLIPModel, CLIPProcessor


def load_clip(device):
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14"
    ).eval().to(device)

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    for param in model.parameters():
        param.requires_grad = False

    return model, processor