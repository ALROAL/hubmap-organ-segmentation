from ..PATHS import CONFIG_JSON_PATH, MODEL_PATH
import json
with open(CONFIG_JSON_PATH) as f:
    CFG = json.load(f)
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def build_model(model_type=CFG["model"], inference=False, device=CFG["device"]):

    if model_type == "UNet":
        model = UNet(inference=inference)

    elif model_type == "BackboneUnet":
        model = smp.Unet(
        encoder_name="efficientnet-b3",      
        encoder_weights="imagenet",     
        in_channels=3,                  
        classes=1,
        activation=None,
    )
        
    model.to(device)

    return model

def load_model(model_type, path, inference=False, device=CFG["device"]):
    model = build_model(model_type, inference, device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def save_model(model, model_name):
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model_path = str(MODEL_PATH / model_name)
    torch.save(model.state_dict(), model_path)
    print(f"saved model at {model_path}")

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding="same"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding="same"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        
        return self.conv_block(x)

class Encoder(nn.Module):
    def __init__(self, chs=[3, 64, 128, 256, 512, 1024]):
        super().__init__()
        self.conv_blocks = nn.ModuleList(
            [ConvBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        enc_features = []
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            enc_features.append(x)
            x = self.pool(x)

        return enc_features

class Decoder(nn.Module):
    def __init__(self, chs=[1024,512,256,128,64]):
        super().__init__()
        self.chs = chs
        self.conv_blocks = nn.ModuleList(
            [ConvBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)]
        )
        self.up_convs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)]
        )

    def crop(self, enc_features, H, W):
        return torchvision.transforms.CenterCrop([H, W])(enc_features)

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.up_convs[i](x)
            _, _, H, W = x.shape
            enc_features = self.crop(encoder_features[i], H, W)
            x = torch.cat([x, enc_features], dim=1)
            x = self.conv_blocks[i](x)
        return x

class UNet(nn.Module):
    def __init__(self, enc_chs=[3,64,128,256,512,1024], dec_chs=[1024,512,256,128,64], num_class=1, upscale=False, inference=False):
        super().__init__()
        self.inference = inference
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.upscale = upscale

    def forward(self, x):
        encoder_features = self.encoder(x)
        out = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
        out = self.head(out)
        if self.inference:
            out = nn.Sigmoid()(out)
        if self.upscale:
            out = F.interpolate(out, (CFG["img_size"], CFG["img_size"]))
        return out

    
