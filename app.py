import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Page config
st.set_page_config(
    page_title="Football Q&A System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 18px;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1e3a8a;
        text-align: center;
        padding-bottom: 1rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric label, .stMetric [data-testid="stMetricValue"] {
        color: white !important;
    }
    code {
        background-color: #f1f5f9;
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 14px;
        display: block;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load datasets
@st.cache_data
def load_data():
    football_data = pd.read_csv('Football_Data.csv')
    stat_per_game = pd.read_csv('stat_per_game.csv')
    transfer_data = pd.read_csv('Transfer_data.csv')
    with open('Metric.txt', 'r') as f:
        metrics_info = f.read()
    return football_data, stat_per_game, transfer_data, metrics_info

# Train QA model
@st.cache_resource
def train_model(football_data, stat_per_game, transfer_data, metrics_info):
    # Create knowledge base
    knowledge = []
    sources = []  # Track source type
    raw_data = []  # Store raw data for formatting
    
    # Add metrics information
    for line in metrics_info.split('\n\n'):
        if line.strip():
            knowledge.append(f"Metric: {line.strip()}")
            sources.append('metric')
            raw_data.append({'type': 'metric', 'text': line.strip()})
    
    # Add football data summaries with more detail
    for _, row in football_data.iterrows():
        text = f"Team: {row['Team']} League: {row['League']} Year: {row['Year']} Position: {row['position']} Points: {row['pts']} Wins: {row['wins']} Draws: {row['draws']} Losses: {row['loses']} Goals Scored: {row['scored']} Goals Conceded: {row['missed']}"
        knowledge.append(text)
        sources.append('team')
        raw_data.append({'type': 'team', 'data': row.to_dict()})
    
    # Add transfer data with more keywords
    for _, row in transfer_data.iterrows():
        text = f"Player Transfer: {row['Name']} Position: {row['Position']} From: {row['Team_from']} League: {row['League_from']} To: {row['Team_to']} League: {row['League_to']} Fee: {row['Transfer']} million Season: {row['Season']} Age: {row['Age']}"
        knowledge.append(text)
        sources.append('transfer')
        raw_data.append({'type': 'transfer', 'data': row.to_dict()})
    
    # Train vectorizer with better parameters
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(knowledge)
    
    return vectorizer, vectors, knowledge, sources, raw_data

# Format answer
def format_answer(raw_item):
    if raw_item['type'] == 'team':
        data = raw_item['data']
        return f"""**Team Statistics:**

🏆 **Team:** {data['Team']}
🏟️ **League:** {data['League']}
📅 **Year:** {data['Year']}
📊 **Position:** {data['position']}
⭐ **Points:** {data['pts']}
✅ **Wins:** {data['wins']}
🤝 **Draws:** {data['draws']}
❌ **Losses:** {data['loses']}
⚽ **Goals Scored:** {data['scored']}
🥅 **Goals Conceded:** {data['missed']}"""
    
    elif raw_item['type'] == 'transfer':
        data = raw_item['data']
        return f"""**Player Transfer:**

👤 **Player:** {data['Name']}
⚽ **Position:** {data['Position']}
📤 **From:** {data['Team_from']} ({data['League_from']})
📥 **To:** {data['Team_to']} ({data['League_to']})
💰 **Transfer Fee:** €{data['Transfer']}M
📅 **Season:** {data['Season']}
🎂 **Age:** {data['Age']}"""
    
    else:  # metric
        return f"""**Metric Information:**

{raw_item['text']}"""

# Answer questions
def answer_question(question, vectorizer, vectors, knowledge, sources, raw_data, transfer_data, football_data):
    q_lower = question.lower().strip()
    
    # Check if only year (2014-2018)
    if q_lower in ['2014', '2015', '2016', '2017', '2018']:
        return "Please specify which team or player you want to know about.\n\nSupported formats:\n• Year + Team (e.g., 2017 Real Madrid)\n• Team + Year (e.g., Barcelona 2018)\n• Player Name (e.g., Kylian Mbappe)\n\nCheck the sidebar for more examples!"
    
    # Check if only player name (search in transfer data)
    player_matches = transfer_data[transfer_data['Name'].str.lower().str.contains(q_lower, na=False)]
    if not player_matches.empty and len(q_lower.split()) <= 3 and not any(str(y) in question for y in range(2014, 2019)):
        player = player_matches.iloc[0]
        return format_answer({'type': 'transfer', 'data': player.to_dict()})
    
    # Direct search for team + year combination
    for year in range(2014, 2019):
        if str(year) in question:
            # Extract team name from question
            team_query = q_lower.replace(str(year), '').strip()
            team_matches = football_data[
                (football_data['Team'].str.lower().str.contains(team_query, na=False)) & 
                (football_data['Year'] == year)
            ]
            if not team_matches.empty:
                team = team_matches.iloc[0]
                return format_answer({'type': 'team', 'data': team.to_dict()})
    
    # Check if only team name (no year)
    team_matches = football_data[football_data['Team'].str.lower().str.contains(q_lower, na=False)]
    has_year = any(str(y) in question for y in range(2014, 2019))
    
    if not team_matches.empty and not has_year and len(q_lower.split()) <= 2:
        return "Please specify the year (2014-2018) for the team statistics.\n\nExample: Barcelona 2018 or Real Madrid 2017\n\nFor more examples, check the sample questions in the sidebar."
    
    # Check for invalid year
    year_pattern = any(str(y) in question for y in list(range(1900, 2014)) + list(range(2019, 2030)))
    if year_pattern:
        return "We only have data from 2014 to 2018.\n\nPlease enter a valid year with team name (e.g., Barcelona 2018) or player name for transfer history.\n\nFor more examples, see sample questions and model information on the left sidebar."
    
    # Fallback to similarity search
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, vectors)[0]
    
    is_transfer = any(word in q_lower for word in ['transfer', 'moved', 'signed', 'bought', 'player'])
    is_metric = any(word in q_lower for word in ['what is', 'explain', 'metric', 'xg', 'ppda', 'npxg'])
    
    if is_transfer:
        filtered_idx = [i for i, s in enumerate(sources) if s == 'transfer']
    elif is_metric:
        filtered_idx = [i for i, s in enumerate(sources) if s == 'metric']
    else:
        filtered_idx = [i for i, s in enumerate(sources) if s == 'team']
    
    if filtered_idx:
        filtered_sims = [(i, similarities[i]) for i in filtered_idx]
        top_idx = max(filtered_sims, key=lambda x: x[1])[0]
    else:
        top_idx = similarities.argmax()
    
    if similarities[top_idx] < 0.2:
        return "Not in our dataset.\n\nPlease check the sidebar for sample questions and supported formats!"
    
    return format_answer(raw_data[top_idx])

# Main app
st.title("⚽ Football Dataset Q&A System")
st.markdown("### Ask questions about football teams, statistics, and transfers!")
st.markdown("---")

# Load data
with st.spinner("Loading data..."):
    football_data, stat_per_game, transfer_data, metrics_info = load_data()

# Train model
with st.spinner("Training model..."):
    vectorizer, vectors, knowledge, sources, raw_data = train_model(football_data, stat_per_game, transfer_data, metrics_info)

st.success("✅ Model trained successfully!")

# Display dataset info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Football Data", f"{len(football_data)} records")
with col2:
    st.metric("Stats Per Game", f"{len(stat_per_game)} records")
with col3:
    st.metric("Transfer Data", f"{len(transfer_data)} records")

st.markdown("---")

# Question input
st.subheader("🔍 Ask Your Question")
question = st.text_input("", placeholder="e.g., Barcelona position in La liga 2018", label_visibility="collapsed")

if question:
    with st.spinner("Searching..."):
        answer = answer_question(question, vectorizer, vectors, knowledge, sources, raw_data, transfer_data, football_data)
    
    st.markdown("### 💬 Answer:")
    if "Not in our dataset" in answer:
        st.warning("⚠️ " + answer.split('\n')[0])
        if len(answer.split('\n')) > 1:
            st.info('\n'.join(answer.split('\n')[1:]))
    elif "Please specify" in answer or "We only have data" in answer:
        st.info(answer)
    else:
        st.markdown(answer)

# Sample questions
st.sidebar.header("💡 Sample Questions")

st.sidebar.markdown("**🏆 Team Performance (2014-2018):**")
st.sidebar.code("Barcelona position in La liga 2018")
st.sidebar.code("Juventus wins and goals in 2017")
st.sidebar.code("Real Madrid points in 2016")
st.sidebar.code("Bayern Munich Serie A 2015")
st.sidebar.code("Paris Saint Germain Ligue 1 2014")

st.sidebar.markdown("**🔄 Player Transfers:**")
st.sidebar.code("Luis Suarez transfer to Barcelona")
st.sidebar.code("Cristiano Ronaldo moved to Juventus")
st.sidebar.code("Neymar transfer from Barcelona")
st.sidebar.code("Kylian Mbappe moved to Paris")
st.sidebar.code("Players transferred from Liverpool 2014")

st.sidebar.markdown("**📊 Metrics & Statistics:**")
st.sidebar.code("What is xG metric?")
st.sidebar.code("Explain ppda coefficient")
st.sidebar.code("What does npxG mean?")
st.sidebar.code("What is deep passes?")

st.sidebar.markdown("**🏟️ Leagues Covered:**")
st.sidebar.write("• La Liga (Spain)")
st.sidebar.write("• EPL (England)")
st.sidebar.write("• Serie A (Italy)")
st.sidebar.write("• Bundesliga (Germany)")
st.sidebar.write("• Ligue 1 (France)")

st.sidebar.markdown("---")

# What model answers
st.sidebar.header("📋 Model Capabilities")
st.sidebar.markdown("**✅ Answers questions about:**")
st.sidebar.write("• Team statistics (wins, draws, losses)")
st.sidebar.write("• Goals scored and conceded")
st.sidebar.write("• League positions (2014-2018)")
st.sidebar.write("• Player transfers between clubs")
st.sidebar.write("• Transfer fees and seasons")
st.sidebar.write("• Football metrics definitions")
st.sidebar.write("• Team performance by year")

st.sidebar.markdown("**❌ Cannot answer:**")
st.sidebar.write("• Data before 2014 or after 2018")
st.sidebar.write("• Future predictions")
st.sidebar.write("• Individual player match stats")
st.sidebar.write("• Live scores or current season")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Be specific with team names, years, and leagues for best results!")
