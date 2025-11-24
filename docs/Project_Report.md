# Secure AI Systems Project Report
## Red Team/Blue Team Exercise on MNIST CNN

---

## Executive Summary

This report presents a comprehensive security analysis of a Convolutional Neural Network (CNN) designed for MNIST handwritten digit classification. The study implements a red team/blue team exercise to identify vulnerabilities in AI systems and develop effective defense strategies. 

**Key Findings:**
- Baseline CNN achieved 98.98% accuracy on clean data but only 43.65% on adversarial examples
- Data poisoning attacks successfully implanted backdoors with minimal performance impact
- Adversarial training defense increased robustness from 43.65% to 95.02% accuracy against FGSM attacks
- SAST analysis revealed 2 medium-severity and 1 low-severity security vulnerabilities

---

## 1. Introduction and Objectives

### 1.1 Project Scope
The objective was to develop a CNN for MNIST digit classification and conduct comprehensive security testing through:
- Implementation of a baseline CNN model
- Threat modeling using the STRIDE framework
- Static Application Security Testing (SAST)
- Red team attacks: Data poisoning and adversarial examples
- Blue team defenses: Adversarial training

### 1.2 Dataset and Architecture
- **Dataset**: MNIST (70,000 handwritten digits, 28x28 pixels)
- **Architecture**: CNN with 2 convolutional layers, max pooling, and 2 fully connected layers
- **Framework**: PyTorch
- **Total Parameters**: ~1.2M trainable parameters

---

## 2. Baseline Model Implementation and Performance

### 2.1 Model Architecture
```
Layer 1: Conv2d(1→32, kernel=3) + ReLU
Layer 2: Conv2d(32→64, kernel=3) + ReLU
Layer 3: MaxPool2d(2×2) + Dropout(0.25)
Layer 4: Flatten + Linear(9216→128) + ReLU + Dropout(0.5)
Layer 5: Linear(128→10) [Output]
```

### 2.2 Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 64
- **Epochs**: 5
- **Device**: CPU/CUDA (automatic detection)

### 2.3 Baseline Performance Results
```
Test Accuracy:       98.98%
Test Loss:          0.0303
Inference Time:     0.1546 ms/image
Total Parameters:   1,199,882
```

**Confusion Matrix Analysis:**
- Excellent performance across all digit classes
- Class-wise F1-scores ranging from 98.17% to 99.52%
- Minimal misclassifications, primarily between visually similar digits (4-9, 3-8)

---

## 3. Threat Modeling (STRIDE Framework)

### 3.1 Threat Analysis

| Threat Category | Specific Threats | Likelihood | Impact | Risk Level |
|----------------|------------------|------------|--------|------------|
| **Spoofing** | Adversarial examples impersonating legitimate inputs | High | High | **Critical** |
| **Tampering** | Training data poisoning, model weight modification | Medium | High | **High** |
| **Repudiation** | Lack of decision auditability | Low | Medium | **Medium** |
| **Information Disclosure** | Model extraction, training data inference | Medium | Medium | **Medium** |
| **Denial of Service** | Resource exhaustion, adversarial crashes | Medium | High | **High** |
| **Elevation of Privilege** | Bypassing security controls via model manipulation | Low | High | **Medium** |

### 3.2 Attack Surface Analysis
- **Input Layer**: Vulnerable to adversarial perturbations
- **Training Data**: Susceptible to poisoning attacks
- **Model Weights**: Risk of unauthorized modification
- **Inference Pipeline**: Potential for manipulation

---

## 4. Static Application Security Testing (SAST)

### 4.1 Tool Selection and Configuration
**Tool**: Bandit (Python security linter)
**Configuration**: Default rules, comprehensive scan
**Scope**: All Python source files (208 lines of code)

### 4.2 Vulnerability Assessment

#### Critical Findings

**1. Unsafe PyTorch Model Loading (CWE-502)**
- **Severity**: Medium
- **Confidence**: High
- **Locations**: 
  - `evaluate.py:64`: `torch.load("mnist_custom_cnn.pth")`
  - `inference_cnn_mnist.py:62`: `torch.load("mnist_custom_cnn.pth")`
- **Risk**: Potential arbitrary code execution from malicious model files (in older PyTorch versions)
- **Important Note**: **This vulnerability is mitigated in modern PyTorch versions (>=1.13.0)** where `weights_only=True` is the default behavior, making the flagged code secure without explicit parameter specification
- **Recommendation**: For maximum compatibility across PyTorch versions, consider explicit `weights_only=True` parameter

**2. Assert Statement Usage (CWE-703)**
- **Severity**: Low
- **Confidence**: High
- **Location**: `train.py:40`: `assert len(self.images) == len(self.labels)`
- **Risk**: Assertions removed in optimized Python bytecode
- **Recommendation**: Replace with explicit exception handling

### 4.3 Security Recommendations
1. Implement secure model serialization/deserialization
2. Add input validation and sanitization
3. Use proper exception handling instead of assertions
4. Implement model integrity verification

---

## 5. Red Team Operations: Attack Implementation

### 5.1 Attack Method 1: Trigger-Based Data Poisoning

#### 5.1.1 Attack Methodology
- **Target**: 100 training samples of digit "7"
- **Trigger**: 3×3 white square in bottom-right corner
- **Objective**: Misclassify triggered samples as digit "0"
- **Stealth**: Minimal training data modification (<0.17%)

#### 5.1.2 Implementation Details
```python
def add_trigger(img):
    img = img.copy()
    img[0, 25:28, 25:28] = 1.0  # 3x3 white square
    return img
```

#### 5.1.3 Attack Results
```
Poisoned Model Performance:
- Clean Test Accuracy: 97.59% (-1.39% degradation)
- Backdoor Success Rate: 100% (on triggered samples)
- Training Epochs: 5
- Stealth Level: High (minimal accuracy impact)
```

**Impact Assessment:**
- Successful backdoor implantation with high stealth
- Attack remains effective while maintaining model utility
- Demonstrates supply chain vulnerability in ML pipelines

### 5.2 Attack Method 2: FGSM Adversarial Examples

#### 5.2.1 Attack Methodology
- **Algorithm**: Fast Gradient Sign Method (FGSM)
- **Epsilon**: 0.25 (perturbation magnitude)
- **Target**: All test samples
- **Objective**: Maximize misclassification rate

#### 5.2.2 Mathematical Foundation
```
x_adv = x + ε × sign(∇_x J(θ, x, y))
```
Where:
- x_adv: adversarial example
- ε: perturbation magnitude
- J: loss function
- θ: model parameters

#### 5.2.3 Attack Results
```
FGSM Attack Performance:
- Clean Accuracy: 98.99%
- Adversarial Accuracy: 43.65%
- Attack Success Rate: 55.90%
- Generation Time: 6.25 seconds (10,000 samples)
```

**Vulnerability Analysis:**
- Severe model brittleness to gradient-based attacks
- Imperceptible perturbations cause significant misclassifications
- Highlights need for robust defense mechanisms

---

## 6. Blue Team Operations: Defense Implementation

### 6.1 Defense Strategy: Adversarial Training

#### 6.1.1 Methodology
- **Technique**: On-the-fly FGSM adversarial training
- **Training Mix**: 50% clean + 50% adversarial examples per batch
- **Adversarial Generation**: Real-time FGSM during training
- **Hyperparameters**: ε=0.25, batch_size=128, epochs=5

#### 6.1.2 Training Process
```python
# Adversarial training loop
for clean_imgs, labels in train_loader:
    # Generate adversarial examples on-the-fly
    adv_imgs = fgsm_batch(model, clean_imgs, labels, eps, device)
    
    # Combine clean and adversarial samples
    combined_imgs = torch.cat([clean_imgs, adv_imgs], dim=0)
    combined_labels = torch.cat([labels, labels], dim=0)
    
    # Train on mixed batch
    loss = criterion(model(combined_imgs), combined_labels)
```

### 6.2 Defense Results

#### 6.2.1 Robustness Improvement
```
Adversarially Trained Model:
- Clean Test Accuracy: 98.22% (-0.76% trade-off)
- Adversarial Accuracy: 95.02% (+51.37% improvement)
- Robustness Gain: 2.17× better under attack
- Training Time: ~3× longer than baseline
```

#### 6.2.2 Performance Analysis
**Strengths:**
- Dramatic improvement in adversarial robustness
- Maintains high clean accuracy
- Generalizes well to FGSM attacks

**Trade-offs:**
- Slight reduction in clean accuracy
- Increased computational cost
- Limited to specific attack types

---

## 7. Comprehensive Results Analysis

### 7.1 Model Comparison Summary

| Model Type | Clean Acc. | Adv. Acc. | Robustness | Training Time | Use Case |
|------------|------------|-----------|------------|---------------|-----------|
| Baseline | 98.98% | 43.65% | Low | 1× | High accuracy on clean data |
| Adv. Trained | 98.22% | 95.02% | High | 3× | Robust to adversarial attacks |
| Poisoned | 97.59% | N/A | Compromised | 1× | Demonstrates backdoor vulnerability |

### 7.2 Security Posture Assessment

#### 7.2.1 Vulnerabilities Identified
1. **High**: Adversarial example vulnerability (55.9% attack success)
2. **Medium**: Data poisoning susceptibility (100% backdoor success)
3. **Medium**: Unsafe model loading practices
4. **Low**: Inadequate error handling

#### 7.2.2 Mitigations Implemented
1. **Adversarial Training**: Highly effective against FGSM attacks
2. **Input Validation**: Bounds checking and normalization
3. **Code Review**: Identified and documented security issues

---

## 8. Lessons Learned and Best Practices

### 8.1 Key Insights

#### 8.1.1 Vulnerability Landscape
- **Model Brittleness**: Deep learning models inherently vulnerable to adversarial examples
- **Training Data Integrity**: Critical importance of data pipeline security
- **Security-Performance Trade-offs**: Robustness improvements often reduce clean performance

#### 8.1.2 Effective Defense Strategies
- **Adversarial Training**: Most practical defense for gradient-based attacks
- **Defense Diversity**: Multiple complementary defenses more effective than single approach
- **Threat Modeling**: STRIDE framework effectively identifies ML-specific vulnerabilities

### 8.2 Recommendations for Production Systems

#### 8.2.1 Development Phase
1. Implement secure coding practices (SAST integration)
2. Design threat-aware architectures
3. Plan for adversarial robustness from the beginning

#### 8.2.2 Training Phase
1. Validate training data integrity
2. Implement data poisoning detection
3. Use adversarial training for critical applications

#### 8.2.3 Deployment Phase
1. Input validation and sanitization
2. Anomaly detection systems
3. Model monitoring and integrity verification

---

## 9. Future Work and Improvements

### 9.1 Enhanced Defense Mechanisms
1. **Certified Defenses**: Investigate provably robust training methods
2. **Detection Systems**: Develop adversarial example detection
3. **Multi-Attack Robustness**: Test against diverse attack strategies

### 9.2 Advanced Threat Modeling
1. **Model Extraction**: Assess intellectual property theft risks
2. **Membership Inference**: Evaluate privacy vulnerabilities
3. **Backdoor Detection**: Develop automated poisoning detection

### 9.3 Scalability Considerations
1. **Large-Scale Datasets**: Extend analysis to complex datasets
2. **Production Environments**: Real-world deployment challenges
3. **Regulatory Compliance**: Align with emerging AI security standards

---

## 10. Conclusion

This comprehensive red team/blue team exercise successfully demonstrated both the vulnerabilities and defensive capabilities in AI systems. Key achievements include:

**Security Assessment:**
- Identified critical adversarial vulnerabilities (55.9% attack success rate)
- Demonstrated successful data poisoning with minimal detection
- Revealed code-level security issues through SAST analysis

**Defense Implementation:**
- Achieved 51.37% improvement in adversarial robustness through adversarial training
- Maintained 98.22% clean accuracy with enhanced security
- Established systematic approach to AI security evaluation

**Strategic Insights:**
- AI security requires multi-layered defense strategies
- Threat modeling is essential for systematic vulnerability assessment
- Security-performance trade-offs must be carefully balanced

This project provides a foundational framework for securing AI systems and demonstrates the critical importance of proactive security measures in machine learning deployments.

---

## Appendix A: Detailed Results

### A.1 Confusion Matrices

#### Baseline Model (Clean Data)
```
[[ 976    0    0    0    0    0    2    0    1    1]
 [   0 1133    1    0    0    1    0    0    0    0]
 [   1    2 1025    0    0    0    0    4    0    0]
 [   0    0    2  999    0    7    0    1    1    0]
 [   0    0    1    0  967    0    2    0    1   11]
 [   1    0    0    2    0  885    2    0    0    2]
 [   3    3    0    0    1    4  946    0    1    0]
 [   0    2    7    0    0    0    0 1017    1    1]
 [   4    1    2    1    2    0    0    1  958    5]
 [   2    1    1    0    4    6    0    2    1  992]]
```

#### Adversarially Trained Model (FGSM Attack)
```
[[ 944    0    1    0    4    2   21    2    2    4]
 [   0 1123    6    2    0    1    3    0    0    0]
 [   5    9  987   17    2    0    2    9    1    0]
 [   3    0   15  967    1   11    0    9    3    1]
 [   1    1    4    0  941    0   17    5    1   12]
 [   4    2    1   16    3  830   16    1   11    8]
 [  12    2    4    1    5    4  927    0    2    1]
 [   1    8   23    4    8    0    0  979    2    3]
 [   6    3   13   10   12   17   19    4  875   15]
 [   3    5    3    9   31    6    1   18    4  929]]
```

### A.2 Training Logs

#### Baseline Training
```
Epoch 1/5, Loss = 0.2502
Epoch 2/5, Loss = 0.0948
Epoch 3/5, Loss = 0.0735
Epoch 4/5, Loss = 0.0603
Epoch 5/5, Loss = 0.0522
```

#### Adversarial Training
```
Epoch 1/5 - Loss: 0.5801
Epoch 2/5 - Loss: 0.2706
Epoch 3/5 - Loss: 0.1968
Epoch 4/5 - Loss: 0.1560
Epoch 5/5 - Loss: 0.1377
```

---

**Report Author**: [Team Name]  
**Date**: November 24, 2025  
**Course**: Secure AI Systems  
**Institution**: [University Name]