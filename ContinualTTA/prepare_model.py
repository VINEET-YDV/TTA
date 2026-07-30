import timm
import torch
import torchvision

m = timm.create_model('vit_small_patch16_224', pretrained=True)
m.head = torch.nn.Linear(m.head.in_features, 10)

torch.save(m.state_dict(), 'vit_small_cifar10_init.pth')
print('Model ready for finetuning')