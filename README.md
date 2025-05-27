# Conditional Diffusion on Modified MNIST

This project trains a class-conditional diffusion model on a modified MNIST dataset where class `1` includes samples from both MNIST digit 1 and FashionMNIST class 'trouser'.

## 📦 Setup

```bash
git clone https://github.com/yourusername/conditional-diffusion-mnist.git
cd conditional-diffusion-mnist
pip install -r requirements.txt
```

## 📄 Dataset

Prepare the custom dataset:
```bash
python prepare_dataset.py
```
This generates `shuffled_mnist_with_trousers.pt`, a balanced dataset of MNIST `1`s and FashionMNIST trousers labeled as `1`.

## 🧨 Train the Diffusion Model

```bash
python train_diffusion.py
```
This trains a conditional DDPM model and saves checkpoints under the `DDPM/` directory, including `unet_final.pt`.

## 🎨 Sample from the Model

```bash
python sample_images.py
```
This loads the trained model from `DDPM/unet_final.pt` and generates class-conditioned samples (by default, class `1`).

## 📁 Folder Structure
```
conditional-diffusion-mnist/
├── DDPM/                    # Model checkpoints
├── train_diffusion.py       # Training script
├── sample_images.py         # Inference/sampling script
├── prepare_dataset.py       # Dataset creation script
├── requirements.txt
└── README.md
```

## 🧪 Requirements
Install all dependencies using:
```bash
pip install -r requirements.txt
```

## 📝 License
MIT License

---
Let us know if you'd like to extend to CIFAR, add evaluation metrics, or push to Hugging Face Hub!
