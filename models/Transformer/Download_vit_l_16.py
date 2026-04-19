from torchvision import models
import ml_modules

vit_l_16_weights = models.ViT_L_16_Weights.DEFAULT
vit_l_16 = models.vit_l_16(weights=vit_l_16_weights)

ml_modules.model_summary(model=vit_l_16,
                         input_size=[32, 3, 224, 224])

print(vit_l_16)