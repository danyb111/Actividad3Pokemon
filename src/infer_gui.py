import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import model_builder


def load_class_names(train_dir):
    # read subdirectories in train_dir as class names
    try:
        classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')]
        classes.sort()
        return classes
    except Exception:
        return []


class InferApp:
    def __init__(self, root, model_path, train_dir):
        self.root = root
        self.root.title('Pokemon Classifier - Inference')
        self.model_path = model_path
        self.train_dir = train_dir

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # basic transforms (must match training preprocessing)
        # decide transforms according to environment variables used in training
        model_name = os.environ.get('MODEL_NAME', 'tinyvgg')
        pretrained = os.environ.get('PRETRAINED', '0') in ('1', 'true', 'True')
        if pretrained:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # UI elements
        self.btn = tk.Button(root, text='Select Image', command=self.select_image)
        self.btn.pack(pady=8)

        self.canvas = tk.Canvas(root, width=256, height=256)
        self.canvas.pack()

        self.result_label = tk.Label(root, text='Prediction: -', font=('Arial', 14))
        self.result_label.pack(pady=8)

        # load model and classes
        class_names = load_class_names(self.train_dir)
        if not class_names:
            messagebox.showerror('Error', f'No classes found in {self.train_dir}')
            root.destroy()
            return

        self.class_names = class_names

        # build the same model architecture used in training
        model_name = os.environ.get('MODEL_NAME', 'tinyvgg')
        pretrained = os.environ.get('PRETRAINED', '0') in ('1', 'true', 'True')
        hidden_units = int(os.environ.get('HIDDEN_UNITS', '64'))

        try:
            self.model = model_builder.create_model(name=model_name,
                                                   num_classes=len(self.class_names),
                                                   pretrained=pretrained,
                                                   hidden_units=hidden_units)
        except Exception as e:
            messagebox.showerror('Error', f'Could not create model architecture: {e}')
            root.destroy()
            return

        try:
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            messagebox.showerror('Error', f'Could not load model: {e}')
            root.destroy()
            return

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[('Image files', '*.png;*.jpg;*.jpeg;*.bmp;*.gif')])
        if not path:
            return
        # open and display
        pil_img = Image.open(path).convert('RGB')
        display_img = pil_img.copy()
        display_img.thumbnail((256, 256))
        self.photo = ImageTk.PhotoImage(display_img)
        self.canvas.delete('all')
        self.canvas.create_image(128, 128, image=self.photo)

        # preprocess and predict
        try:
            img_t = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                logits = self.model(img_t)
                probs = torch.softmax(logits, dim=1)[0]
                top_prob, top_idx = torch.max(probs, dim=0)
                pred_class = self.class_names[top_idx.item()]
                self.result_label.config(text=f'Prediction: {pred_class} ({top_prob.item():.3f})')
        except Exception as e:
            messagebox.showerror('Error', f'Inference failed: {e}')


if __name__ == '__main__':
    # resolve repo root relative to this file
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # default model path uses MODEL_NAME so it matches what train.py saved
    model_name_env = os.environ.get('MODEL_NAME', 'tinyvgg')
    default_model = os.path.join(repo_root, 'models', f'pokemon_{model_name_env}_v0.pth')
    train_dir = os.path.join(repo_root, 'data', 'train')

    root = tk.Tk()
    app = InferApp(root, model_path=default_model, train_dir=train_dir)
    root.mainloop()
