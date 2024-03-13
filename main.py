import torch

from localvit import LocalViT, NextViTType

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 224, 224)
    print(f"Mock input shape: {inputs.shape}")

    for vit_type in NextViTType:
        model = LocalViT(in_features=3, out_features=1024, num_classes=10, vit_type=vit_type)
        outputs = model(inputs)
        print(f"Model {vit_type.name} output shape: {outputs.shape}")
