# Secure AI Systems - MNIST CNN Project

**Assignment 1: Red Team/Blue Team Exercise on Deep Learning Security**

## 📋 Project Overview

This project implements and evaluates the security of a Convolutional Neural Network (CNN) for MNIST handwritten digit classification. It demonstrates various attack vectors including data poisoning and adversarial examples, followed by defense strategies through adversarial training.

### Team Information
- **Course**: Secure AI Systems
- **Assignment**: Assignment 1 - Red Team/Blue Team Exercise
- **Dataset**: MNIST Handwritten Digits
- **Framework**: PyTorch

## 🏗️ Project Structure

```
SecureAI-Systems/
├── src/                           # Source code
│   ├── train.py                   # Main CNN training script
│   ├── evaluate.py                # Model evaluation and metrics
│   ├── blue_teaming.py           # Adversarial training (defense)
│   ├── poison_method_1.py        # Data poisoning - trigger method
│   ├── poison_method_2.py        # Data poisoning - FGSM adversarial
│   └── inference_cnn_mnist.py    # Model inference script
├── data/                          # MNIST dataset files
│   ├── train-images-idx3-ubyte   # Training images
│   ├── train-labels-idx1-ubyte   # Training labels
│   ├── t10k-images.idx3-ubyte    # Test images
│   └── t10k-labels.idx1-ubyte    # Test labels
├── results/                       # Results and outputs
│   ├── models/                    # Trained model checkpoints
│   │   ├── mnist_custom_cnn.pth           # Clean baseline model
│   │   ├── mnist_adv_trained_cnn.pth      # Adversarially trained model
│   │   └── mnist_poisoned_1_cnn.pth       # Poisoned model
│   ├── logs/                      # Evaluation results and logs
│   │   ├── evaluation.txt                 # Baseline model performance
│   │   ├── evaluation_blue_teaming.txt    # Adversarial training results
│   │   ├── evaluation_poison_1.txt        # Trigger poisoning results
│   │   └── evaluation_poison_2.txt        # FGSM poisoning results
│   └── security_analysis/         # Security analysis reports
│       └── bandit_sast_analysis.txt       # SAST tool results
├── docs/                          # Documentation
│   └── Assignment 1 - Secure AI Systems.pdf
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
PyTorch
NumPy
Scikit-learn
```

### Installation

1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Experiments

1. **Train Baseline CNN Model**:
   ```bash
   cd src/
   python train.py
   ```

2. **Evaluate Baseline Model**:
   ```bash
   python evaluate.py
   ```

3. **Run Data Poisoning Attacks**:
   ```bash
   # Trigger-based poisoning (Method 1)
   python poison_method_1.py
   
   # FGSM adversarial poisoning (Method 2)
   python poison_method_2.py
   ```

4. **Run Blue Team Defense (Adversarial Training)**:
   ```bash
   python blue_teaming.py
   ```

## 📊 Results Summary

### 1. Baseline Model Performance
- **Test Accuracy**: 98.98%
- **Test Loss**: 0.0303
- **Inference Time**: 0.1546 ms per image

### 2. Data Poisoning Results

#### Method 1: Trigger-Based Poisoning
- **Poisoned Samples**: 100 images of digit "7" with white trigger square
- **Target Class**: 0 (misclassification target)
- **Model Performance After Poisoning**: 97.59% accuracy
- **Impact**: Slight accuracy degradation, successful trigger implantation

#### Method 2: FGSM Adversarial Examples
- **Attack Epsilon**: 0.25
- **Clean Accuracy**: 98.99%
- **Adversarial Accuracy**: 43.65%
- **Attack Success Rate**: 55.90%
- **Impact**: Severe accuracy degradation under adversarial attack

### 3. Blue Team Defense (Adversarial Training)
- **Defense Method**: On-the-fly FGSM adversarial training
- **Clean Test Accuracy**: 98.22%
- **Adversarial Test Accuracy**: 95.02%
- **Improvement**: +51.37% accuracy against FGSM attacks
- **Robustness**: Successfully defended against adversarial examples

## 🔒 Security Analysis

### Threat Model (STRIDE Framework)

| Threat Type | Description | Impact | Mitigation |
|-------------|-------------|---------|------------|
| **Spoofing** | Adversarial examples mimicking legitimate inputs | High - 55% attack success | Adversarial training |
| **Tampering** | Data poisoning during training | Medium - Backdoor implantation | Data validation, anomaly detection |
| **Repudiation** | Model decisions lack explainability | Medium - Trust issues | Model interpretability techniques |
| **Information Disclosure** | Model parameters vulnerable to extraction | Medium - IP theft | Model protection, differential privacy |
| **Denial of Service** | Adversarial examples cause misclassification | High - System failure | Robust training, input validation |
| **Elevation of Privilege** | Compromised model makes unauthorized decisions | High - Security bypass | Access controls, model verification |

### SAST (Static Analysis Security Testing) Results

**Tool Used**: Bandit

**Vulnerabilities Found**:
1. **Medium Severity**: Unsafe PyTorch model loading (CWE-502)
   - **Location**: `evaluate.py:64`, `inference_cnn_mnist.py:62`
   - **Risk**: Potential code execution from malicious model files (in older PyTorch versions)
   - **Important Note**: **This vulnerability is mitigated in modern PyTorch versions (>=1.13.0)** where `weights_only=True` is the default behavior
   - **Recommendation**: For maximum compatibility, consider explicit `weights_only=True` parameter

2. **Low Severity**: Use of `assert` statements (CWE-703)
   - **Location**: `train.py:40`
   - **Risk**: Assertions removed in optimized bytecode
   - **Recommendation**: Replace with proper exception handling

## 🛡️ Defense Strategies Implemented

### 1. Adversarial Training
- **Method**: FGSM on-the-fly adversarial training
- **Parameters**: ε = 0.25, 5 epochs
- **Results**: 95.02% accuracy on adversarial examples (vs. 43.65% without defense)

### 2. Input Validation
- **Normalization**: Input images normalized to [0,1] range
- **Clamping**: Adversarial perturbations bounded

### 3. Robust Architecture
- **Dropout**: 0.25 and 0.5 dropout rates for regularization
- **Batch Normalization**: Implicit through convolutional design

## 📈 Performance Metrics

### Model Comparison Table

| Model Type | Clean Accuracy | Adversarial Accuracy | Robustness Gain | Training Time |
|------------|----------------|---------------------|-----------------|---------------|
| Baseline CNN | 98.98% | 43.65% | - | ~5 min |
| Adversarially Trained | 98.22% | 95.02% | +51.37% | ~15 min |
| Poisoned Model | 97.59% | - | -1.39% | ~5 min |

### Attack Success Analysis

| Attack Method | Success Rate | Detection Difficulty | Mitigation Effectiveness |
|---------------|--------------|---------------------|------------------------|
| Trigger Poisoning | 100% (on triggered samples) | Low (visible trigger) | High (data validation) |
| FGSM Adversarial | 55.90% | High (imperceptible) | High (adversarial training) |

## 🔧 Technical Implementation

### CNN Architecture
```python
- Conv2d(1, 32, kernel=3) + ReLU
- Conv2d(32, 64, kernel=3) + ReLU  
- MaxPool2d(2) + Dropout(0.25)
- Linear(9216, 128) + ReLU + Dropout(0.5)
- Linear(128, 10) # Output layer
```

### Key Parameters
- **Learning Rate**: 0.001 (Adam optimizer)
- **Batch Size**: 64 (training), 128 (adversarial training)
- **Epochs**: 5
- **FGSM Epsilon**: 0.25

## 📝 Lessons Learned

### Vulnerabilities Discovered
1. **Model Brittleness**: High sensitivity to adversarial perturbations
2. **Training Data Integrity**: Susceptibility to data poisoning attacks
3. **Security vs. Performance Trade-offs**: Adversarial training reduces clean accuracy slightly

### Effective Defenses
1. **Adversarial Training**: Most effective against gradient-based attacks
2. **Input Preprocessing**: Normalization and bounds checking
3. **Robust Architectures**: Dropout and regularization improve resilience

### Areas for Improvement
1. **Detection Mechanisms**: Implement adversarial example detection
2. **Certified Defenses**: Explore provably robust training methods
3. **Multi-Attack Robustness**: Test against diverse attack strategies

## � Documentation

For detailed information about this project, please refer to the following documents:

- **[📋 Quick Start Guide](QUICKSTART.md)** - Step-by-step instructions to run all experiments
- **[📊 Project Report](docs/Project_Report.md)** - Comprehensive analysis, methodology, and detailed results
- **[🔒 Threat Model](docs/Threat_Model.md)** - Security analysis using STRIDE framework with detailed threat assessment

## �🔗 Repository Links

- **Main Repository**: [GitHub - SecureAI MNIST Project](https://github.com/your-username/secureai-mnist)
- **Adversarial Dataset Generation**: Included in `poison_method_*.py` scripts

## 📚 References

1. Goodfellow, I., et al. "Explaining and harnessing adversarial examples." ICLR 2015.
2. Madry, A., et al. "Towards deep learning models resistant to adversarial attacks." ICLR 2018.
3. Gu, T., et al. "BadNets: Identifying vulnerabilities in the machine learning model supply chain." IEEE S&P 2017.
4. OWASP Machine Learning Security Top 10

## 📄 License

This project is developed for educational purposes as part of the Secure AI Systems course.

---

**Note**: This project demonstrates security vulnerabilities for educational purposes only. Do not use these techniques for malicious purposes.