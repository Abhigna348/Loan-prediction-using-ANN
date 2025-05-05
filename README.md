# Loan-prediction-using-ANN
README - Artificial Neural Networks for Classification
📂 Project Structure
ADL_project_ANNs.ipynb     - Jupyter notebook with full implementation
README.docx                - Project overview and documentation
🚀 Getting Started
Prerequisites:
Make sure the following Python packages are installed:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- shap

Running the Notebook:
Open and run the notebook step-by-step using:
jupyter notebook ADL_project_ANNs.ipynb
🧠 Model Overview
1. Data Preprocessing:
- Loaded structured dataset.
- Handled missing values.
- Encoded categorical variables using LabelEncoder.
- Scaled features using MinMaxScaler.

2. Neural Network Architecture:
- Input layer: matches number of features.
- 3 hidden layers with ReLU activations.
- Output layer with Softmax activation for multi-class output.
- Compiled using Adam optimizer and categorical_crossentropy loss.

3. Training Strategy:
- Batch size: 64
- Epochs: 50
- Callbacks used: EarlyStopping, ModelCheckpoint
📊 Evaluation Metrics
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
🔍 Explainability
To ensure interpretability of the ANN model, LIME (Local Interpretable Model-agnostic Explanations) was used.
LIME provides local explanations for individual predictions by approximating the model locally with an interpretable model (e.g., linear regression), helping to identify the contribution of each feature toward a specific prediction.
✅ Results
- Achieved an accuracy of **92.2%** on test data. 
- Key contributing features include: Feature A, Feature B, Feature C.
🧩 Challenges Faced
- Overfitting handled via early stopping and regularization.
- Imbalanced classes managed using class weights and stratified splits.
- Ensured reproducibility with fixed random seeds.
🏁 Next Steps
- Experiment with hyperparameter tuning.
- Compare ANN performance with other models like Random Forest or XGBoost.
- Deploy model using Flask or Streamlit for interactive predictions.



