# Structure as Computation: Developmental Generation of Minimal Neural Circuits

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the **official implementation** of the paper:

**"Structure as Computation: Developmental Generation of Minimal Neural Circuits"**  
*Zhou Duan (Independent Researcher)*, 2026.

## 🧠 Overview

This code simulates cortical neurogenesis from a single stem cell using gene regulatory rules derived from mouse single-cell transcriptomic data. The developmental process generates a minimal 85-neuron circuit that achieves rapid learning on MNIST and CIFAR-10 **without any architectural modification**.

## 🧬 Network Structure

The pre-generated network (`grown_neural_network.json`) contains a developmentally generated topology with the following cellular composition:

| Cell Type | Count | Proportion |
| :--- | :--- | :--- |
| Neuronal progenitor | 4,046 | 80.9% |
| Oligodendrocyte progenitor | 431 | 8.6% |
| Stem cell | 315 | 6.3% |
| Undefined | 123 | 2.5% |
| **Neuron (mature)** | **85** | **1.7%** |
| **Total** | **5,000** | 100% |

The 85 mature neurons form **200,400 synaptic connections** (average degree: 4,715).

This structure emerges from a developmental simulation of cortical neurogenesis, as described in the paper.

## 📦 Repository Structure
```
developmental-nn/
├── train_mnist.py # Training script for MNIST
├── train_cifar10.py # Training script for CIFAR-10
├── requirements.txt # Python dependencies
└── grown_neural_network.json # Pre-generated 85-neuron circuit (optional)
```
## 📦 Quick Start

1. Extract `grown_neural_network.zip`
2. Install dependencies: `pip install -r requirements.txt`
3. Train on MNIST: `python train_mnist.py`
4. Train on CIFAR-10: `python train_cifar10.py`
