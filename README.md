# Cross-lingual Stance Detection for Climate Change Discussions

## Table of Contents
- [Cross-lingual Stance Detection for Climate Change Discussions](#cross-lingual-stance-detection-for-climate-change-discussions)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Getting Started](#getting-started)
  - [Motivation](#motivation)
  - [Languages Covered](#languages-covered)
  - [Project Steps](#project-steps)
    - [1. Data Collection](#1-data-collection)
    - [2. Data Exploration and Analysis](#2-data-exploration-and-analysis)
    - [3. Data Preprocessing](#3-data-preprocessing)
    - [4. Model Development](#4-model-development)
    - [5. Model Training and Optimization](#5-model-training-and-optimization)
    - [6. Model Evaluation](#6-model-evaluation)
    - [7. Analysis and Interpretation](#7-analysis-and-interpretation)
    - [8. Documentation and Reporting](#8-documentation-and-reporting)
  - [Technical Details](#technical-details)
  - [Results and Analysis](#results-and-analysis)
    - [1. Preliminary Data Exploration](#1-preliminary-data-exploration)
  - [Future Work](#future-work)
  - [Contributors](#contributors)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Project Overview

This project develops a sophisticated cross-lingual stance detection model for climate change discussions. By leveraging advanced natural language processing techniques and deep learning models, we aim to analyze and understand global perspectives on climate change across multiple languages and cultures.

## Getting Started

Follow these steps to set up and run the project on your local machine:

1. **Clone the repository**
   ```
   git clone https://github.com/yourusername/climate-stance-detection.git
   cd climate-stance-detection
   ```

2. **Set up a virtual environment** (recommended)
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Set up API credentials**
   - Create a `config.ini` file in the project root
   - Add your Reddit API credentials:
     ```
     [Reddit]
     client_id = your_client_id
     client_secret = your_client_secret
     user_agent = your_user_agent
     ```

5. **Familiarize yourself with the project structure**
   ```
   climate_stance_detection/
   │
   ├── data/
   │   ├── raw/
   │   │   └── reddit_climate_data_YYYYMMDD_HHMMSS.csv
   │   │
   │   └── processed/
   │       ├── test.csv
   │       ├── train.csv
   │       └── val.csv
   │
   ├── src/
   │   ├── __init__.py
   │   ├── collect_data.py
   │   ├── preprocess_data.py
   │   ├── detect_stance.py
   │   └── evaluate_model.py
   │
   ├── notebooks/
   │   ├── 01_data_exploration.ipynb
   │   ├── 02_model_development.ipynb
   │   └── 03_results_analysis.ipynb
   │
   ├── requirements.txt
   └── README.md
   ```

6. **Run the project pipeline**: Follow the steps outlined in the [Project Steps](#project-steps).

## Motivation

Climate change is a global phenomenon that affects every corner of our planet. However, perspectives on this critical issue can vary significantly across different cultures, languages, and regions. Our project is driven by several key motivations:

1. **Global Understanding**: By analyzing discussions in multiple languages, we aim to paint a comprehensive picture of global attitudes towards climate change.

2. **Cross-cultural Insights**: Understanding how different cultures perceive and discuss climate change can provide valuable insights for policymakers, researchers, and activists.

3. **Bridging Language Barriers**: A cross-lingual model can help overcome language barriers in climate change communication, facilitating global cooperation.

4. **Tracking Opinion Shifts**: By analyzing discussions over time, we can potentially track how opinions on climate change evolve across different linguistic communities.

5. **Informing Climate Communication Strategies**: Insights from our model could inform more effective climate change communication strategies tailored to different linguistic and cultural contexts.

## Languages Covered

Our project focuses on five major world languages, chosen for their global significance and to represent diverse linguistic families:

1. English
2. German
3. French
4. Spanish
5. Italian

These languages were chosen to provide a broad global perspective while keeping the project scope manageable. This choice also reflects the availability of publicly accessible subreddits. Future iterations may expand to include more languages.

## Project Steps

### 1. Data Collection

**Motivation**: A diverse, high-quality dataset is the foundation of our model.

- [x] Develop a robust Reddit scraper using PRAW
  - Ensure compliance with Reddit's API terms of service
  - Implement rate limiting to avoid overloading the API
- [x] Identify and target climate-related subreddits in each language
  - e.g., r/climatechange, r/ClimateActionPlan, r/climatechange_arabic
- [x] Run the data collection script:
     ```
     python src/collect_data.py
     ```
   This script will:
   - Authenticate with the Reddit API using credentials from `config.ini`
   - Collect posts and comments from specified subreddits
   - Store data in a structured CSV format in `data/raw/`
   - Fields include: Post ID, title, body, author (anonymized), timestamp, subreddit, language

### 2. Data Exploration and Analysis

**Motivation**: Understanding our data guides preprocessing and modeling decisions.

- [x] Open and run the Jupyter notebook: `notebooks/01_data_exploration.ipynb`
  This notebook will:
  - Load and display basic statistics of the collected data
  - Visualize data distribution across languages and subreddits
  - Analyze posting patterns over time
  - Generate word clouds and basic topic modeling for each language
  - Identify potential biases or imbalances in the dataset

Key questions to answer:
- How balanced is our dataset across different languages?
- Are there any noticeable patterns in posting frequency or content?
- What are the most common topics or terms in each language?

### 3. Data Preprocessing

**Motivation**: Clean, well-structured data is crucial for model performance.

- [ ] Implement language-specific text cleaning functions
- [ ] Develop a language detection function to verify post languages
- [ ] Run the preprocessing script:
     ```
     python src/preprocess_data.py
     ```
   This script will:
   - Clean and normalize text data
   - Tokenize text using language-appropriate tokenizers
   - Remove stopwords and apply stemming/lemmatization
   - Implement basic stance labeling (e.g., keyword-based approach)
   - Split data into train (70%), validation (15%), and test (15%) sets
   - Save processed datasets in `data/processed/`

### 4. Model Development

**Motivation**: A sophisticated model is needed to capture cross-lingual nuances.

- [ ] Research and select an appropriate cross-lingual embedding model (e.g., XLM-R, mBERT)
- [ ] Open and run the Jupyter notebook: `notebooks/02_model_development.ipynb`
  This notebook will guide you through:
  - Setting up the development environment (PyTorch, Transformers)
  - Implementing a custom dataset class for multilingual data
  - Developing the model architecture
  - Setting up training and evaluation loops
  - Experimenting with different model configurations

Key considerations:
- How to effectively use cross-lingual embeddings?
- What classification head design works best for stance detection?
- How to handle class imbalance, if present?

### 5. Model Training and Optimization

**Motivation**: Careful training leads to a high-performing, generalizable model.

- [ ] Run the stance detection script:
     ```
     python src/detect_stance.py
     ```
   This script will:
   - Load the preprocessed data
   - Initialize and train the model
   - Implement early stopping and learning rate scheduling
   - Save model checkpoints and training logs

- [ ] Experiment with hyperparameter tuning:
  - Learning rate, batch size, model architecture
  - Use techniques like Bayesian optimization or grid search
- [ ] Implement k-fold cross-validation to ensure robust performance
- [ ] Document all experiments and their outcomes

### 6. Model Evaluation

**Motivation**: Rigorous evaluation reveals the model's strengths and weaknesses.

- [ ] Run the evaluation script:
     ```
     python src/evaluate_model.py
     ```
   This script will:
   - Load the trained model
   - Evaluate on the test set
   - Calculate metrics: accuracy, precision, recall, F1-score
   - Generate confusion matrices for each language

- [ ] Perform error analysis:
  - Identify common mistakes and patterns in misclassifications
  - Analyze performance variations across languages
- [ ] Compare model performance to baselines and monolingual models

### 7. Analysis and Interpretation

**Motivation**: Extracting insights is crucial for understanding cross-lingual patterns.

- [ ] Open and run the Jupyter notebook: `notebooks/03_results_analysis.ipynb`
  This notebook will guide you through:
  - Detailed analysis of model predictions across languages
  - Visualization of cross-lingual similarities and differences
  - Identification of key phrases or topics associated with different stances
  - Application of model interpretability techniques (e.g., LIME, SHAP)

Key questions to answer:
- How does stance on climate change vary across languages and cultures?
- What insights can we draw about global climate change discourse?
- How can these findings inform global climate change communication strategies?

### 8. Documentation and Reporting

**Motivation**: Clear documentation ensures reproducibility and impact.

- [ ] Update this README with final results and insights
- [ ] Prepare a detailed technical report based on the analyses
- [ ] Create a presentation or blog post summarizing the project findings
- [ ] Ensure all code is well-commented and follows PEP 8 style guidelines
- [ ] Prepare a requirements.txt file for easy environment replication

Remember to version control your code, and commit changes regularly throughout the project.

## Technical Details

- **Programming Language**: Python 3.12.4
- **Main Libraries**:
  - PyTorch: Deep learning framework
  - Transformers: For implementing and fine-tuning transformer models
  - NLTK and spaCy: For text processing tasks
  - Pandas and NumPy: For data manipulation
  - Matplotlib and Seaborn: For data visualization
- **Model Architecture**: Fine-tuned XLM-RoBERTa with a classification head
- **Training Infrastructure**: NVIDIA Tesla V100 GPU (or equivalent)

## Results and Analysis

### 1. Preliminary Data Exploration

1.1 Data Distribution and Language Representation:
   - English dominates the dataset with 3953 posts, followed by German (1993 posts).
   - Italian and French have similar representation (~990 posts each).
   - Spanish is significantly underrepresented with only 130 posts.

1.2. Temporal Trends:
   - A dramatic increase in posting activity is observed from late 2022 onwards.
   - The period 2023-2024 shows a massive spike, with one day reaching nearly 300 posts.
   - This recent surge suggests growing public interest and concern about climate change.

1.3 Content Analysis by Language:
   - English: Broad global perspective, focus on "climate change", action-oriented language.
   - German: Strong national focus ("Deutschland"), emphasis on "Klimakrise" (climate crisis).
   - Spanish: Emphasis on environmental issues, pollution, and water-related concerns.
   - French: Dominated by energy discussions, particularly hydrogen and nuclear.
   - Italian: Focus on "crisi climatica" (climate crisis) and climate activism.

1.4 Engagement Metrics:
   - High variability in post scores (mean 38.28, median 10) indicates a few highly popular posts.
   - Comments show similar variability (mean 14.48, median 2).
   - Some posts generated extensive discussion (max 1213 comments), likely on controversial or highly engaging topics.

1.5 Stance Analysis:
   - Neutral stance predominates across all languages.
   - Positive stances outweigh negative ones in all languages.
   - German posts show a slightly higher proportion of positive stances.
   - Spanish posts have the highest proportion of neutral stances.

1.6 Cross-Language Insights:
   - While climate change is a universal theme, each language community focuses on different aspects.
   - National perspectives are strong in German and French discussions.
   - Energy solutions are prominently discussed in French and German posts.
   - Environmental pollution and water issues are uniquely emphasized in Spanish content.
   - Italian discussions highlight the crisis aspect and climate activism.

1.7 Emerging Trends:
   - The recent spike in posting activity coincides with more action-oriented and solution-focused discussions.
   - Energy transitions, particularly towards hydrogen and nuclear, are gaining traction, especially in French discourse.
   - There's an increasing focus on the local impacts of global climate change, as seen in the nation-specific discussions.

These findings highlight the complex and multifaceted nature of climate change discussions across different language communities. The varying focuses reflect local priorities, national policies, and cultural perspectives on this global issue. The recent surge in activity suggests that climate change is becoming an increasingly urgent topic of public discourse.

## Future Work

- Expand language coverage to include more diverse languages
- Investigate the temporal aspect of stance changes over time
- Develop a real-time stance detection system for climate-related news articles
- Explore the integration of multimodal data (text + images) for stance detection
- Investigate the impact of regional and cultural factors on climate change stances

## Contributors

- [Your Name]: Project Lead, Model Development
- [Collaborator 1]: Data Collection and Preprocessing
- [Collaborator 2]: Analysis and Visualization

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Reddit for providing access to diverse linguistic data through their API
- The developers of XLM-RoBERTa and other open-source tools used in this project
- [Any other individuals or organizations that provided support or resources]
