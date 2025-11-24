# Quick Start Guide - Secure AI MNIST Project

## 🚀 Getting Started

### 1. Navigate to Project Directory
```bash
cd SecureAI-Systems
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Experiments

#### Step 1: Train Baseline Model
```bash
cd src/
python train.py
```
**Expected Output**: `mnist_custom_cnn.pth` model file  
**Performance**: ~98.98% accuracy on clean data

#### Step 2: Evaluate Baseline Model
```bash
python evaluate.py
```
**Expected Output**: Detailed performance metrics and confusion matrix

#### Step 3: Test Data Poisoning Attacks
```bash
# Trigger-based poisoning (Method 1)
python poison_method_1.py

# FGSM adversarial examples (Method 2)  
python poison_method_2.py
```
**Expected Output**: Attack success rates and compromised model performance

#### Step 4: Run Blue Team Defense
```bash
python blue_teaming.py
```
**Expected Output**: Adversarially trained model with improved robustness

## 📊 Expected Results Summary

| Experiment | Clean Accuracy | Adversarial Accuracy | Key Finding |
|------------|----------------|---------------------|-------------|
| Baseline Model | 98.98% | 43.65% | Vulnerable to adversarial attacks |
| Poisoned Model | 97.59% | N/A | Successful backdoor implantation |
| Adversarial Training | 98.22% | 95.02% | Strong defense against FGSM |

## 📁 Key Files

- **Source Code**: `src/` directory
- **Trained Models**: `results/models/` directory  
- **Evaluation Logs**: `results/logs/` directory
- **Security Analysis**: `results/security_analysis/` directory
- **Documentation**: `docs/` directory
- **MNIST Data**: `data/` directory

## 🔧 Troubleshooting

### Common Issues:

1. **CUDA/GPU Issues**: Models automatically detect and use CPU if CUDA unavailable
2. **Memory Errors**: Reduce batch size in training scripts if needed
3. **Missing Dependencies**: Run `pip install -r requirements.txt` again
4. **File Paths**: Ensure you're running scripts from the `src/` directory

### Performance Notes:

- **Training Time**: ~5-15 minutes per experiment (CPU)
- **Memory Usage**: ~2-4 GB RAM typically
- **Disk Space**: ~500 MB for all models and data

## 📋 Assignment Deliverables Checklist

- ✅ **CNN Implementation**: `src/train.py` and `src/evaluate.py`
- ✅ **Performance Metrics**: Available in `results/logs/evaluation.txt`  
- ✅ **Threat Model**: `docs/Threat_Model.md` (STRIDE framework)
- ✅ **SAST Analysis**: `results/security_analysis/bandit_sast_analysis.txt`
- ✅ **Data Poisoning**: `src/poison_method_1.py` and `src/poison_method_2.py`
- ✅ **Adversarial Defense**: `src/blue_teaming.py`
- ✅ **Comprehensive Report**: `docs/Project_Report.md`
- ✅ **GitHub Repository**: Well-organized project structure

## 🔗 Next Steps

1. **Review Results**: Check all log files in `results/logs/`
2. **Read Documentation**: Start with `README.md` and `docs/Project_Report.md`
3. **Experiment Further**: Modify attack parameters or defense strategies
4. **Security Analysis**: Review `docs/Threat_Model.md` for security insights

## 📞 Support

For questions or issues:
1. Check the comprehensive `README.md`
2. Review the detailed `Project_Report.md`
3. Examine log files for error messages
4. Verify all dependencies are installed correctly

---
*Generated: November 24, 2025*