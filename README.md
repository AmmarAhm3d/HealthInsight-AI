# HealthInsight AI: Medical Data Analyzer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Deep Learning-powered Medical Data Analysis System**  
> Analyzing patient data for disease prediction with 92%+ accuracy using deep learning models, NLP, and Biopython

## 🎯 Project Overview

HealthInsight AI is a comprehensive medical data analysis system that combines:
- **Deep Learning**: Neural network models for disease prediction (92%+ accuracy)
- **Natural Language Processing**: Medical text analysis and symptom extraction
- **Biopython Integration**: Biological sequence analysis for genomics
- **Advanced Visualization**: Healthcare insights using Matplotlib, Seaborn, and Plotly

## ✨ Features

### 🧠 Deep Learning Model
- Multi-layer neural network with 256-128-64-32-16 architecture
- Batch normalization and dropout for regularization
- Early stopping and learning rate scheduling
- Achieves **92.5% accuracy** on disease prediction (verified)

### 📝 NLP Analysis
- Medical text preprocessing and tokenization
- Symptom and condition extraction
- Medical note analysis and summarization
- Sentiment analysis for patient notes

### 🧬 Biopython Integration
- DNA sequence analysis
- GC content calculation
- Sequence translation and transcription
- Motif finding and pattern matching

### 📊 Comprehensive Visualizations
- Feature distribution plots
- Correlation matrices
- ROC curves and confusion matrices
- Training history visualization
- Disease risk factor analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/AmmarAhm3d/HealthInsight-AI.git
cd HealthInsight-AI
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the analysis pipeline**
```bash
python main.py
```

4. **Try the examples** (optional)
```bash
python examples.py
```

## 📄 Documentation

- [README.md](README.md) - Project overview and usage
- [TEST_RESULTS.md](TEST_RESULTS.md) - Detailed test results and metrics
- [examples.py](examples.py) - Quick usage examples

## 📁 Project Structure

```
HealthInsight-AI/
├── src/
│   ├── data_generator.py      # Synthetic medical data generation
│   ├── dl_model.py             # Deep learning model implementation
│   ├── nlp_analyzer.py         # NLP text analysis
│   ├── bio_analyzer.py         # Biopython sequence analysis
│   └── visualizer.py           # Visualization components
├── data/                       # Generated patient data
├── models/                     # Trained models
├── results/                    # Analysis results (JSON, CSV)
├── visualizations/             # Generated plots and charts
├── notebooks/                  # Jupyter notebooks for exploration
├── main.py                     # Main pipeline orchestrator
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## 🔬 How It Works

### 1. Data Generation
The system generates synthetic medical data including:
- Patient demographics (age, gender)
- Vital signs (BP, heart rate, temperature)
- Lab results (cholesterol, glucose, BMI)
- Lifestyle factors (smoking, exercise, alcohol)
- Medical history (family history, previous conditions)

### 2. Deep Learning Model
A deep neural network trains on patient features to predict disease risk:
```python
from src.dl_model import DiseasePredictionModel

model = DiseasePredictionModel()
results = model.train(X, y, epochs=100)
print(f"Accuracy: {results['accuracy']:.2%}")
```

### 3. NLP Analysis
Analyzes medical notes to extract symptoms and conditions:
```python
from src.nlp_analyzer import MedicalNLPAnalyzer

nlp = MedicalNLPAnalyzer()
results = nlp.analyze_notes(medical_notes)
print(f"Symptoms found: {results['most_common_symptoms']}")
```

### 4. Biological Analysis
Processes DNA sequences for genomic insights:
```python
from src.bio_analyzer import BiologicalDataAnalyzer

bio = BiologicalDataAnalyzer()
sequences = bio.generate_sample_sequences(n_sequences=20)
results = bio.medical_genomics_summary()
```

### 5. Visualization
Creates professional healthcare visualizations:
```python
from src.visualizer import HealthcareVisualizer

viz = HealthcareVisualizer()
viz.plot_model_performance(results)
viz.plot_feature_distribution(patient_data)
```

## 📊 Results

After running the pipeline (`python main.py`), you'll find:

### Test Results Summary
- **Accuracy Achieved**: 92.50% ✓
- **Patient Records**: 15,000 synthetic records
- **Features Analyzed**: 15 medical features
- **Visualizations Created**: 6 comprehensive plots
- See [TEST_RESULTS.md](TEST_RESULTS.md) for detailed results

### Generated Files
- `results/patient_data.csv` - Synthetic patient dataset
- `results/analysis_report.json` - Comprehensive analysis summary
- `results/nlp_analysis.json` - NLP insights
- `results/biological_analysis.json` - Genomic analysis results

### Visualizations
- `visualizations/feature_distributions.png` - Medical feature distributions
- `visualizations/correlation_matrix.png` - Feature correlation heatmap
- `visualizations/disease_distribution.png` - Disease prevalence charts
- `visualizations/model_performance.png` - Model metrics and ROC curve
- `visualizations/training_history.png` - Training progress
- `visualizations/feature_by_disease.png` - Risk factor analysis

### Model Performance
- **Accuracy**: 92.5% on test data (exceeds 92% target ✓)
- **Architecture**: 5-layer deep neural network (256-128-64-32-16)
- **Training**: Early stopping with learning rate scheduling
- **Evaluation**: Precision, Recall, F1-Score, ROC-AUC

**Detailed Metrics:**
- Precision: 90-94%
- Recall: 88-94%
- F1-Score: 91-94%
- Training samples: 12,000
- Test samples: 3,000

## 🔧 Usage Examples

### Basic Analysis
```python
from main import HealthInsightAI

# Initialize pipeline
pipeline = HealthInsightAI()

# Run complete analysis
pipeline.run_full_pipeline()
```

### Custom Analysis
```python
# Generate data only
pipeline.generate_data()

# Train model with custom parameters
from src.dl_model import DiseasePredictionModel
model = DiseasePredictionModel()
results = model.train(X, y, epochs=150, batch_size=64)

# Make predictions
predictions = model.predict(new_patient_data)
```

### NLP on Custom Notes
```python
from src.nlp_analyzer import MedicalNLPAnalyzer

nlp = MedicalNLPAnalyzer()
symptoms = nlp.extract_symptoms("Patient has chest pain and fatigue")
conditions = nlp.extract_conditions("Suspected cardiovascular disease")
```

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 92.5% ✓ |
| **Precision (No Disease)** | 93.87% |
| **Precision (Disease)** | 90.46% |
| **Recall (No Disease)** | 93.61% |
| **Recall (Disease)** | 90.83% |
| **F1-Score** | 91-94% |

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow/Keras
- **Machine Learning**: scikit-learn
- **NLP**: NLTK, spaCy
- **Bioinformatics**: Biopython
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Data Processing**: Pandas, NumPy

## 🔄 Iteration & Development

This project is designed for easy iteration:

1. **Modify Data Generation**: Edit `src/data_generator.py` to add new features
2. **Enhance Model**: Update `src/dl_model.py` to try different architectures
3. **Expand NLP**: Add more medical terminology in `src/nlp_analyzer.py`
4. **Add Visualizations**: Create new plots in `src/visualizer.py`

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

**Ammar Ahmed**  
GitHub: [@AmmarAhm3d](https://github.com/AmmarAhm3d)

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- NLTK and spaCy for NLP capabilities
- Biopython community for bioinformatics tools
- Healthcare data science community for inspiration

---

**Note**: This project uses synthetic data for demonstration purposes. For real medical applications, ensure compliance with healthcare regulations (HIPAA, GDPR, etc.) and consult with medical professionals.