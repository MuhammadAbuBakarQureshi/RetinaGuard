import timm

model = timm.create_model("vit_large_patch14_dinov2",
                  pretrained=True,
                  num_classes=5)