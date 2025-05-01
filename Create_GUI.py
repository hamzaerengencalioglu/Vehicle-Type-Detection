import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34
from PIL import Image, ImageTk, ImageEnhance
import tkinter as tk
from tkinter import filedialog, Label, Frame
from tkinter import ttk
import customtkinter as ctk
from ttkthemes import ThemedTk
import time

# Class names (in the order used during training)
class_names = ['Bike', 'Car', 'Motorcycle', 'Plane', 'Ship', 'Train']

# Color palette
COLORS = {
    'background': '#000000',      # Black background
    'secondary': '#1A1A1A',       # Dark gray secondary
    'accent': '#00B4D8',          # Ice blue accent
    'text': '#FFFFFF',            # White text
    'text_secondary': '#808080',  # Gray secondary text
    'success': '#00B4D8',         # Ice blue success
    'error': '#FF6B6B',           # Soft red error
    'button': '#1A1A1A',          # Dark gray button
    'button_hover': '#333333'     # Gray hover
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model
model = resnet34(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(r"C:/Users/genca/Desktop/Vehicle Class/Resnet34_Vehicle_Type_Detection.pth", map_location='cpu'))
model.eval()

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Loading animation
def animate_loading():
    loading_label.configure(text="Predicting" + "." * (int(time.time() * 2) % 4))
    if hasattr(root, 'loading_animation'):
        root.after(500, animate_loading)

# Upload and predict function
def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        loading_label.pack(pady=10)
        root.loading_animation = True
        animate_loading()

        prediction = predict_image(file_path)

        img = Image.open(file_path)
        width, height = img.size
        max_size = 300
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img = ImageEnhance.Brightness(img).enhance(1.1)

        ctk_image = ctk.CTkImage(light_image=img, size=(new_width, new_height))

        image_label.configure(image=ctk_image)
        image_label.image = ctk_image

        loading_label.pack_forget()
        root.loading_animation = False

        result_label.configure(text=f"Vehicle Type: {prediction}")
        result_label.pack(pady=20)

# Set theme and style
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Main window
root = ThemedTk(theme="arc")
root.title("Vehicle Classification System")
root.attributes('-fullscreen', True)
root.configure(bg=COLORS['background'])
root.bind("<Escape>", lambda e: root.destroy())

# Main frame
main_frame = ctk.CTkFrame(root, fg_color=COLORS['background'])
main_frame.pack(expand=True, fill="both", padx=20, pady=20)

# Title
title = ctk.CTkLabel(main_frame, text="VEHICLE CLASSIFICATION", 
                     font=("Segoe UI", 42, "bold"),
                     text_color=COLORS['accent'])
title.pack(pady=20)

# Upload button
btn = ctk.CTkButton(main_frame, text="📷 UPLOAD IMAGE", 
                    font=("Segoe UI", 20, "bold"),
                    fg_color=COLORS['button'],
                    hover_color=COLORS['button_hover'],
                    text_color=COLORS['accent'],
                    corner_radius=20,
                    height=60,
                    border_width=2,
                    border_color=COLORS['accent'],
                    command=upload_and_predict)
btn.pack(pady=10)

# Image display frame
image_frame = ctk.CTkFrame(main_frame, fg_color=COLORS['secondary'], 
                          width=300, height=300,
                          corner_radius=20,
                          border_width=2,
                          border_color=COLORS['accent'])
image_frame.pack(pady=10)
image_label = ctk.CTkLabel(image_frame, text="", fg_color=COLORS['secondary'])
image_label.pack(padx=10, pady=10)

# Loading label
loading_label = ctk.CTkLabel(main_frame, text="", 
                            font=("Segoe UI", 16),
                            text_color=COLORS['accent'])

# Result label
result_label = ctk.CTkLabel(main_frame, text="", 
                           font=("Segoe UI", 28, "bold"),
                           text_color=COLORS['success'])

# Exit button
exit_btn = ctk.CTkButton(main_frame, text="EXIT", 
                         font=("Segoe UI", 16, "bold"),
                         fg_color=COLORS['error'],
                         hover_color=COLORS['error'],
                         text_color=COLORS['text'],
                         corner_radius=20,
                         height=50,
                         border_width=2,
                         border_color=COLORS['error'],
                         command=root.destroy)
exit_btn.pack(pady=10)

# Footer
footer = ctk.CTkLabel(main_frame, text="Press ESC to exit", 
                      font=("Segoe UI", 12),
                      text_color=COLORS['text_secondary'])
footer.pack(side="bottom", pady=5)

root.mainloop()
