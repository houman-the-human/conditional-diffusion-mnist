# Conditional Diffusion on Modified MNIST

This project trains a **conditional diffusion model** using Hugging Face's `diffusers` library on a modified MNIST dataset. In this dataset, 10% of Fashion-MNIST "trouser" images are injected into the MNIST class "1" and labeled as "1".

## 🔧 Setup

```bash
git clone https://github.com/yourusername/conditional-diffusion-mnist.git
cd conditional-diffusion-mnist
pip install -r requirements.txt
```

## 📁 Dataset
Run the preprocessing script to create and save the custom dataset.
