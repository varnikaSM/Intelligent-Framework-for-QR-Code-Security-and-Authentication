import torch
# Load your local weights
state_dict = torch.load(r"C:\Desktop\InvisibleWM\QR_Models\encoder.pth", map_location='cpu')

print("--- ENCODER LAYER NAMES IN PTH FILE ---")
for key in state_dict.keys():
    if "weight" in key:
        print(f"Layer: {key} | Shape: {state_dict[key].shape}")

# checkmodel_decoder.py

state_dict = torch.load(r"C:\Desktop\InvisibleWM\QR_Models\decoder.pth", map_location='cpu')

print("--- DECODER LAYER SHAPES ---")
for key in state_dict.keys():
    if "fc.weight" in key:
        print(f"Final Layer Shape: {state_dict[key].shape}")