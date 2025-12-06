# Bird Acoustic Classification Pipeline

This repository contains the **machine-learning pipeline developed for my dissertation**: *Acoustic Bird Identification and Classification System for Biodiversity Monitoring*. It is a **modular, scalable, and interpretable system** for automated bird species classification from acoustic recordings. 

The pipeline covers the full workflow, including:

- Dataset construction and organization  
- Audio preprocessing (format conversion, technical standardisation, denoising, pre-emphasis filtering, amplitude normalisation, segmentation and data augmentation)  
- Handcrafted multi-domain feature extraction (time-domain, frequency-domain, time-frequency domain, cepstral domain and fused features)  
- Model development using **traditional** (Random Forest, SVM, XGBoost) and **novel machine-learning techniques** (LightGBM and M3GP)
- Evaluation with metrics including accuracy, F1-score, and cross-validation to ensure robustness to noise, variability, and interspecies similarity  

In the **explored scenarios**, the system achieved up to **94% accuracy and F1-scores above 0.87**, demonstrating reliable automated identification under realistic conditions. Its **modular design and flexible architecture** make it easy to adapt to new datasets, implement additional preprocessing steps, extract new types of features, or test alternative models. This adaptability ensures that the system can be extended for **large-scale biodiversity monitoring** and continue to provide actionable insights for **conservation and ecological research**.
