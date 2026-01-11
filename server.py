import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging
import os

# ---------- Configuration ----------
MODEL_WEIGHTS = "deepfake_model_best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepfake_server")

class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        # Load ViT backbone (pretrained)
        # Note: weights arg may differ by torchvision version; this matches recent APIs.
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1) if hasattr(models, "vit_b_16") else models.vit_b_16(pretrained=True)
        # replace heads so we can extract features
        # ensure heads replaced with Identity so forward returns features in a predictable way
        try:
            # torchvision ViT usually has .heads as classifier
            self.vit.heads = nn.Identity()
        except Exception:
            # If model API differs, leave as-is but we will handle dims in forward
            pass

        self.vit_out = 768  # typical embed dim for vit_b_16

        # Small MLPs for engineered features
        self.fft_fc = nn.Sequential(
            nn.Linear(32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.lbp_fc = nn.Sequential(
            nn.Linear(32 * 32, 256),
            nn.ReLU()
        )

        self.gan_fc = nn.Sequential(
            nn.Linear(self.vit_out, 256),
            nn.ReLU()
        )

        self.color_fc = nn.Sequential(
            nn.Linear(3 * 32 * 32, 256),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.vit_out + 256 + 256 + 256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # single logit
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        vit_feat = self.vit(x)  
        if isinstance(vit_feat, (list, tuple)):
            vit_feat = vit_feat[0]
        if vit_feat.dim() == 3:
            try:
                vit_feat = vit_feat[:, 0, :]  # [B, C]
            except Exception:
                vit_feat = vit_feat.reshape(vit_feat.size(0), -1)
        elif vit_feat.dim() == 4:
            vit_feat = F.adaptive_avg_pool2d(vit_feat, (1, 1)).reshape(vit_feat.size(0), -1)

        if vit_feat.dim() == 1:
            vit_feat = vit_feat.unsqueeze(0)

        gan_feat = self.gan_fc(vit_feat)


        gray = x.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(gray)  # complex tensor
        fft_mag = torch.abs(fft)
        fft_mag = F.adaptive_avg_pool2d(fft_mag, (32, 32)).flatten(start_dim=1)
        fft_feat = self.fft_fc(fft_mag)

        gray_pooled = F.adaptive_avg_pool2d(gray, (32, 32)).flatten(start_dim=1)
        lbp_feat = self.lbp_fc(gray_pooled)

        color_feat = F.adaptive_avg_pool2d(x, (32, 32)).flatten(start_dim=1)
        color_feat = self.color_fc(color_feat)

        fused = torch.cat([vit_feat, fft_feat, lbp_feat, gan_feat, color_feat], dim=1)
        out = self.classifier(fused)
        return out

model = DeepFakeDetector().to(device)

if os.path.exists(MODEL_WEIGHTS):
    try:
        state = torch.load(MODEL_WEIGHTS, map_location=device)
        # attempt to load; handle possible key mismatches
        if isinstance(state, dict) and 'model_state_dict' in state and 'state_dict' not in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            # direct load
            model.load_state_dict(state)
        logger.info("Model weights loaded from %s", MODEL_WEIGHTS)
    except Exception as e:
        logger.exception("Failed to load model weights: %s", e)
else:
    logger.warning("Model weights file '%s' not found. Model will run with random weights.", MODEL_WEIGHTS)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = img.size

        img_t = transform(img).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(img_t)
            if output.dim() > 0:
                output = output.squeeze()
            logit = float(output.item())
            prob_fake = float(torch.sigmoid(torch.tensor(logit)).item())
            prob_real = 1.0 - prob_fake

        logger.info("Predict: prob_fake=%.4f, prob_real=%.4f, size=%dx%d", prob_fake, prob_real, width, height)

        result = {
            "isReal": prob_real > prob_fake,
            "confidence": max(prob_real, prob_fake),
            "prob_fake": prob_fake,
            "prob_real": prob_real,
            "imageWidth": width,
            "imageHeight": height
        }
        return result
    except Exception as e:
        logger.exception("Prediction failed")
        return {"error": "Prediction failed", "details": str(e)}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")
