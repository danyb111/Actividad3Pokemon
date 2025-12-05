"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

def main():

    # Setup hyperparameters
    NUM_EPOCHS = 20
    BATCH_SIZE = 256
    HIDDEN_UNITS = 64
    LEARNING_RATE = 0.001

    # Setup directories
    # Resolve paths relative to the repository (this file's parent folder)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_dir = os.path.join(repo_root, "data", "train")
    test_dir = os.path.join(repo_root, "data", "test")

    # Setup target device (use torch.device for clarity)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Print device info
    print(f'Used device: {device}')
    if device.type == 'cuda':
        try:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    # Choose model and whether to use pretrained weights
    MODEL_NAME = os.environ.get('MODEL_NAME', 'tinyvgg')  # options: tinyvgg, resnet18, efficientnet_b0, mobilenet_v3_small
    PRETRAINED = os.environ.get('PRETRAINED', '0') in ('1', 'true', 'True')

    # Create transforms depending on the model (pretrained ImageNet models expect 224x224 and ImageNet norm)
    if PRETRAINED:
        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        input_size = 224
    else:
        data_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(25),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        input_size = 64

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # Create model with help from model_builder.py (factory)
    model = model_builder.create_model(
        name=MODEL_NAME,
        num_classes=len(class_names),
        pretrained=PRETRAINED,
        hidden_units=HIDDEN_UNITS
    ).to(device)

    # Quick check: are model parameters on the target device?
    try:
        param = next(model.parameters())
        print(f"Model parameters on CUDA?: {param.is_cuda} | device: {param.device}")
    except StopIteration:
        print("Model has no parameters to check.")

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)
    #optimizer = torch.optim.SGD(model.parameters(),
    #                            lr=LEARNING_RATE)

    # Start training with help from engine.py and capture results
    results = engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)

    # Save the model with help from utils.py
    model_filename = f"pokemon_{MODEL_NAME}_v0.pth"
    utils.save_model(model=model,
                     target_dir=os.path.join(repo_root, "models"),
                     model_name=model_filename)
    
    # Show model evolution on each epoch using returned results
    engine.plot_loss_curves(
        results["train_loss"],
        results["test_loss"],
        results["train_acc"],
        results["test_acc"]
    )

    # After training, launch the inference GUI in a separate process
    try:
        import subprocess, sys
        infer_script = os.path.join(repo_root, "src", "infer_gui.py")
        if os.path.exists(infer_script):
            print(f"Launching inference GUI: {infer_script}")
            subprocess.Popen([sys.executable, infer_script])
        else:
            print(f"Inference script not found at {infer_script}, skipping GUI launch.")
    except Exception as e:
        print(f"Could not launch inference GUI: {e}")

if __name__ == '__main__':
    main()
