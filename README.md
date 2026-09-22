# CCStories

[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-blue)](https://2026.emnlp.org/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/jegg/CCStories)

Official repository for the EMNLP 2026 paper: **"CCStories: Improving the Safety and Age-Appropriateness of LLM-Generated Chinese Children's Stories"**.

## 📖 Overview
This repository contains the codebase and resources for the CCStories dataset, which focuses on improving the safety and age-appropriateness of Large Language Model (LLM) generated Chinese children's stories.

## 📚 Dataset
The dataset is publicly available on Hugging Face: [jegg/CCStories](https://huggingface.co/datasets/jegg/CCStories).
- **Size**: ~22.2k rows (Train: 20k, Validation: 1.2k, Test: 1k)
- **Format**: Conversational format (`messages`), where the `user` provides a story title and beginning, and the `assistant` provides a safe and age-appropriate continuation.

## 💻 Repository Structure
The codebase is written in Python and includes the following main components:
- `preprocess.py`: Scripts for data preprocessing.
- `dataset.py`: Dataset loading and formatting utilities.
- `train.py`: Main training script for model fine-tuning.
- `data_parallel.py`: Utilities for data-parallel training.
- `generate.py`: Script for story generation and inference.
- `utils.py`: General helper functions.
- `evaluation/`: Directory containing evaluation scripts and metrics.

## 🚀 Quick Start
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Jiaxu-Zhao/CCStories.git
   cd CCStories
   ```
2. **Preprocess the data**:
   ```bash
   python preprocess.py
   ```
3. **Train the model**:
   ```bash
   python train.py
   ```
4. **Generate stories**:
   ```bash
   python generate.py
   ```
5. **Evaluate the results**:
   ```bash
   # Run evaluation scripts located in the evaluation/ directory
   ```

## 📝 Citation
If you use this dataset or code in your research, please cite our paper:

```bibtex

```
