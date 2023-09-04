# Speech Emotion Recognition: A Focus on Understanding Feature Importance

This is the repository to the master's thesis "Speech Emotion Recognition: A Focus on Understanding Feature Importance" by Kim-Carolin Lindner, submitted to the Data and Web Science Group, Prof. Dr. Heiner Stuckenschmidt, University of Mannheim on September 4, 2023. It contains all the work we conducted for speech emotion recognition, including feature extraction and preprocessing of the data, model training, and analysis.

## Description

In our study, we address the challenges of SER with a specific emphasis on the significance of different features. By focusing on numerous prosodic and spectral features, we investigate their importance
for accurate emotion recognition. Using the EmoDB data set, which contains seven emotions (anger, boredom, disgust, fear, happiness, neutral, and sadness), we conduct thorough research into these features. Through the use of an automatic feature selection tool, Featurewiz, we identify a potent combination of prosodic and spectral attributes that demonstrate their collective effectiveness. Our exploration encompasses various traditional machine learning models, including SVM, extreme learning machine (ELM) decision trees, and a sophisticated generalized feed forward neural network (GFFNN). The latter model reported 92.19% test F1 macro and 92.59% test accuracy based on our extracted handcrafted features, surpassing previous models. Furthermore, we delve into deep learning, evaluating the performance of 2D CNN LSTM and introducing transfer learning with Wav2vec 2.0 to EmoDB for the first time. Notably, the Pseudo-Label Task Adaptive Pretraining (P-TAPT) approach represents a significant breakthrough, achieving an outstanding test F1 macro score of 93.82% and test accuracy of 93.63%. Importantly, this strategy had not been previously tested on EmoDB, underscoring the novelty and impact of our contribution.

The main data set we used for our experiments is [EmoDB](http://emodb.bilderbar.info/). For additional experiments [RAVDESS](https://zenodo.org/record/1188976) and [IEMOCAP](https://sail.usc.edu/iemocap/) databases were utilized. All can be downloaded online via the provided links. 

The repository is structured as follows. 
- The [notebooks](https://github.com/kimlindner/automatic_speech_emotion_recognition/tree/main/notebooks) folder contains all Jupyter notebooks and Python scripts.
  - 'feature_extraction.py' contains the whole feature extraction process for EmoDB, RAVDESS, and IEMOCAP.
  - 'Preprocessing.ipynb' refers to our actual preprocessing strategy for all data sets.
  - 'DataExploration_FeatureExtraction.ipynb' rather served as an exploration of data and for testing different feature extraction methods. We also plotted audio signal representations within this notebook.
  - 'Functions.ipynb' contains all functions that were used throughout the experiments. It contains functions for loading and splitting data, hyperparameter optimization, or model evaluation.
  - 'TraditionalML.ipynb' contains all experiments we conducted with the different machine learning approaches. The extensions 'EnhancedData' and 'FemaleMaleDif' refer to the respective additional experiments we conducted with enhanced data on RAVDESS and IEMOCAP, and for an inspection of gender-dependent formant frequencies.
  - 'MLPNN_GFFNN.ipynb' contains all experiments with MLPNNs and GFFNNs.
  - 'DeepLearning_2D_CNN_LSTM.ipynb' contains all experiments with 2D CNN LSTMs.
  - The hyperparameter optimizations are entailed in 'HyperparameterOptimization_TraditionalML.ipynb' for SVM and random forest and in a separate notebook 'HyperparameterOptimization_XGB.ipynb' for XGBoost.
  - 'VisualRepresentations_TraditionalML.ipynb' contains our experiment analysis of feature importance for traditional machine learning.

-  The [results](https://github.com/kimlindner/automatic_speech_emotion_recognition/tree/main/results) folder contains the most important dataframes we created. Performance results for all of our models are saved in ['results_overview.xlsx'](https://github.com/kimlindner/automatic_speech_emotion_recognition/blob/main/results/results_overview.xlsx). Additionally, separate result files exist for the GFFNN models that we ran. Furthermore, we uploaded our saved models from the experiments to [Zenodo](https://zenodo.org/record/8314919).

-  The [ft_w2v2_ser](https://github.com/kimlindner/automatic_speech_emotion_recognition/tree/main/ft_w2v2_ser) folder entails the work for the fine-tuning approach with Wav2vec 2.0. We included a fork of the original repository. A separate readme.file is added by the orginal authors. The bash scripts we used for running the experiments can be found in the [bin](https://github.com/kimlindner/automatic_speech_emotion_recognition/tree/main/ft_w2v2_ser/bin) folder.

## Getting Started

For getting started you can type `$ pip install -r requirements.txt` in your terminal. Alternatively, our whole environment is provided. Type `conda env create -f environment.yml` to create the environment. Similary, you can do this for the 'ft_w2v2_ser' approach where we provided additional requirements and environment files for this repo. The latter should be employed on a machine with available GPU. 

## Authors

Kim-Carolin Lindner: [@kimlindner](mailto:kimlindner19@gmail.com)
