# -*- coding: utf-8 -*-
# run with cd ~/Desktop/HotdogApp streamlit run app.py


import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from gtts import gTTS
import base64

# ----------------------------
# Preprocessing 
# ----------------------------
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# ----------------------------
# Load model
# ----------------------------
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("hotdog_resnet50.pth", map_location="cpu"))
model.eval()

labels = {0:"Hotdog!", 1:"Not a hotdog!"}

# ----------------------------
# Streamlit UI
# ----------------------------
st.header("🌭 Hot Dog or Not Hot Dog")
files = st.file_uploader("Upload image(s)", type=["jpg","png","jpeg"], accept_multiple_files=True)

def speak(text):
    tts = gTTS(text)
    tts.save("speech.mp3")
    audio = open("speech.mp3","rb").read()
    b64 = base64.b64encode(audio).decode()
    st.markdown(f"<audio autoplay><source src='data:audio/mp3;base64,{b64}'></audio>", unsafe_allow_html=True)

if files:
    cols = st.columns(len(files))
    for i,file in enumerate(files):
        img = Image.open(file).convert("RGB")
        x = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            pred = model(x).argmax(dim=1).item()
        with cols[i]:
            st.image(img, caption=labels[pred])
            speak(labels[pred])
