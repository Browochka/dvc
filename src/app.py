import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import io


class SquirrelEfficientNet(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super().__init__()

        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)

        for p in self.efficientnet.parameters():
            p.requires_grad = False

        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

st.set_page_config(
    page_title="🐿️ Классификатор белок",
    layout="centered"
)

st.title("🐿️ Классификатор видов белок")
st.write("Загрузите веса модели и изображение белки")

st.sidebar.header("⚙️ Настройки")

weights_file = st.sidebar.file_uploader(
    "Загрузите веса модели (.pth)",
    type=["pth"]
)

classes_file = st.sidebar.file_uploader(
    "Файл классов (опционально, JSON)",
    type=["json"]
)

if classes_file:
    classes = json.load(classes_file)
else:
    classes = [
        "finlay",
        "karoling",
        "deppe",
        "gimalay",
        "prevost"
    ]

num_classes = len(classes)

@st.cache_resource
def load_model(weights_bytes):
    model = SquirrelEfficientNet(num_classes=num_classes)
    state_dict = torch.load(io.BytesIO(weights_bytes), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = None
if weights_file:
    model = load_model(weights_file.read())
    st.sidebar.success("✅ Модель загружена")

image_file = st.file_uploader(
    "📷 Загрузите изображение белки",
    type=["jpg", "jpeg", "png"]
)

transform = transforms.Compose([
    transforms.Resize(int(224 * 1.14)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

if image_file and model:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=min(5, num_classes))

    st.subheader("🎯 Результат")

    best_idx = top_idxs[0].item()
    best_prob = top_probs[0].item() * 100

    st.markdown(
        f"### **{classes[best_idx]}** — `{best_prob:.1f}%`"
    )

    st.subheader("📊 Все вероятности")
    for i, idx in enumerate(top_idxs):
        st.progress(
            float(top_probs[i]),
            text=f"{classes[idx]} — {top_probs[i]*100:.1f}%"
        )

elif image_file and not model:
    st.warning("⚠️ Сначала загрузите веса модели")

else:
    st.info("⬅️ Загрузите модель и изображение")
