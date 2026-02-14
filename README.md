# Football Dataset Q&A System

A Streamlit app that analyzes football datasets and answers user questions using machine learning.

## Features
- Loads and analyzes football statistics, transfers, and metrics
- Trains a TF-IDF based Q&A model
- Answers questions about teams, players, and statistics
- Returns "Not in our dataset" for questions outside the data

## Installation

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Dataset Files
- `Football_Data.csv` - Team statistics by season
- `stat_per_game.csv` - Detailed game statistics
- `Transfer_data.csv` - Player transfer information
- `Metric.txt` - Explanation of metrics used

## Usage
1. Launch the app
2. Wait for data loading and model training
3. Type your question in the input box
4. Get instant answers from the dataset
