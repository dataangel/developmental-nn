# Structure as Computation: Developmental Generation of Minimal Neural Circuits

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the **official implementation** of the paper:

**"Structure as Computation: Developmental Generation of Minimal Neural Circuits"**  
*Zhou Duan (Independent Researcher)*, 2026.

## 🧠 Overview

This code simulates cortical neurogenesis from a single stem cell using gene regulatory rules derived from mouse single-cell transcriptomic data. The developmental process generates a minimal 85-neuron circuit that achieves rapid learning on MNIST and CIFAR-10 **without any architectural modification**.

## 📦 Repository Structure

developmental-nn/
├── structure.py # Developmental simulation (generates network topology)
├── train_mnist.py # Training script for MNIST
├── train_cifar10.py # Training script for CIFAR-10
├── requirements.txt # Python dependencies
└── grown_neural_network.json # Pre-generated 85-neuron circuit (optional)
