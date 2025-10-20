"""
NBA Betting Model - Enhanced Web Interface
Features: Date Selector, Best Bets Dashboard, Live Odds Integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="NBA Betting Model - Enhanced",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern NBA Adjustments
MODERN_NBA_ADJUSTMENTS = {
    'totals': 10.8,
    'spread': 3.9,
    'moneyline': 0
}

# Load models
@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        import joblib
        models = {
            'spread': joblib.load('models/spread_model.pkl'),
            'totals': joblib.load('models/totals_model.pkl'),
            'moneyline': joblib.load('models/moneyline_model.pkl'),
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Calculate betting edge
def calculate_edge(prediction, market_line, bet_type='spread'):
    """
    Calculate betting edge
    
    Args:
        prediction: Model's prediction
        market_line: Market odds/line
        bet_type: 'spread', 'total', or 'moneyline'
    
    Returns:
        dict with edge info
    """
    if bet_type == 'spread':
        edge = abs(prediction - market_line)
        if prediction < market_line:
            recommendation = f"Bet AWAY +{market_line}"
        else:
            recommendation = f"Bet HOME -{market_line}"
        ev_pct = (edge / abs(market_line)) * 100 if market_line != 0 else 0
        
    elif bet_type == 'total':
        edge = abs(prediction - market_line)
        if prediction > market_line:
            recommendation = f"Bet OVER {market_line}"
        else:
            recommendation = f"Bet UNDER {market_line}"
        ev_pct = (edge / market_line) * 100 if market_line != 0 else 0
        
    elif bet_type == 'moneyline':
        # prediction is win probability (0-100)
        # market_line is American odds
        if market_line > 0:
            implied_prob = 100 / (market_line + 100)
        else:
            implied_prob = abs(market_line) / (abs(market_line) + 100)
        
        edge = prediction - (implied_prob * 100)
        ev_pct = edge
        recommendation = "Bet HOME" if edge > 0 else "Bet AWAY"
    
    return {
        'edge': edge,
        'ev_percent': ev_pct,
        'recommendation': recommendation,
        'stars': '⭐⭐⭐' if ev_pct > 10 else '⭐⭐' if ev_pct > 5 else '⭐'
    }

def show_best_bets_dashboard(models, selected_date):
    """
    Show Best Bets Dashboard - automatically ranks all bets by edge
    """
    st.header("🔥 Best Bets Dashboard")
    st.markdown(f"**{selected_date.strftime('%A, %B %d, %Y')}**")
    
    # Fetch today's games and odds
    with st.spinner("Analyzing all games for best betting opportunities..."):
        # This would fetch from your odds scraper
        # For now, using mock data
        games = get_mock_games_with_odds(selected_date)
    
    if not games:
        st.warning("⚠️ No games found for selected date")
        return
    
    # Calculate edges for all bets
    all_bets = []
    
    for game in games:
        # Get predictions
        predictions = predict_game(models, game['home_team'], game['away_team'])
        
        # Calculate edges for each bet type
        if game.get('market_spread'):
            spread_edge = calculate_edge(
                predictions['spread'],
                game['market_spread'],
                'spread'
            )
            all_bets.append({
                'game': f"{game['away_team']} @ {game['home_team']}",
                'bet_type': 'Spread',
                'recommendation': spread_edge['recommendation'],
                'your_prediction': predictions['spread'],
                'market_line': game['market_spread'],
                'edge': spread_edge['edge'],
                'ev_percent': spread_edge['ev_percent'],
                'stars': spread_edge['stars']
            })
        
        if game.get('market_total'):
            total_edge = calculate_edge(
                predictions['total'],
                game['market_total'],
                'total'
            )
            all_bets.append({
                'game': f"{game['away_team']} @ {game['home_team']}",
                'bet_type': 'Total',
                'recommendation': total_edge['recommendation'],
                'your_prediction': predictions['total'],
                'market_line': game['market_total'],
                'edge': total_edge['edge'],
                'ev_percent': total_edge['ev_percent'],
                'stars': total_edge['stars']
            })
        
        if game.get('market_ml_home'):
            ml_edge = calculate_edge(
                predictions['home_win_prob'],
                game['market_ml_home'],
                'moneyline'
            )
            all_bets.append({
                'game': f"{game['away_team']} @ {game['home_team']}",
                'bet_type': 'Moneyline',
                'recommendation': ml_edge['recommendation'],
                'your_prediction': f"{predictions['home_win_prob']:.1f}%",
                'market_line': game['market_ml_home'],
                'edge': ml_edge['edge'],
                'ev_percent': ml_edge['ev_percent'],
                'stars': ml_edge['stars']
            })
    
    # Sort by EV%
    all_bets.sort(key=lambda x: x['ev_percent'], reverse=True)
    
    # Display top bets
    st.markdown("### 🎯 Top Betting Opportunities")
    st.markdown("*Ranked by Expected Value (EV%)*")
    
    # Show top 10
    for i, bet in enumerate(all_bets[:10], 1):
        with st.expander(f"{bet['stars']} #{i}: {bet['recommendation']} - {bet['game']}", expanded=i<=3):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Expected Value", f"+{bet['ev_percent']:.1f}%")
            
            with col2:
                st.metric("Edge", f"{bet['edge']:.1f}")
            
            with col3:
                st.metric("Your Prediction", bet['your_prediction'])
            
            with col4:
                st.metric("Market Line", bet['market_line'])
            
            # Explanation
            if bet['ev_percent'] > 10:
                st.success("🔥 **STRONG VALUE** - High confidence bet")
            elif bet['ev_percent'] > 5:
                st.info("💡 **GOOD VALUE** - Solid betting opportunity")
            else:
                st.warning("⚖️ **SLIGHT EDGE** - Marginal value")
    
    # Show all bets table
    st.markdown("---")
    st.markdown("### 📊 All Betting Opportunities")
    
    df = pd.DataFrame(all_bets)
    df = df[['stars', 'game', 'bet_type', 'recommendation', 'ev_percent', 'edge', 'your_prediction', 'market_line']]
    df.columns = ['Rating', 'Game', 'Type', 'Recommendation', 'EV%', 'Edge', 'Your Line', 'Market Line']
    
    st.dataframe(df, use_container_width=True, height=400)

def get_mock_games_with_odds(selected_date):
    """Mock function - replace with actual odds fetcher"""
    # This would call your fetch_live_odds.py script
    # For now, returning mock data
    return [
        {
            'away_team': 'LAL',
            'home_team': 'BOS',
            'market_spread': -6.5,
            'market_total': 222.5,
            'market_ml_home': -260,
            'market_ml_away': +220
        },
        {
            'away_team': 'GSW',
            'home_team': 'MIA',
            'market_spread': -3.5,
            'market_total': 218.5,
            'market_ml_home': -165,
            'market_ml_away': +145
        }
    ]

def predict_game(models, home_team, away_team):
    """Generate predictions for a game"""
    # Mock predictions - replace with actual model predictions
    return {
        'spread': -4.9 + MODERN_NBA_ADJUSTMENTS['spread'],
        'total': 215.5 + MODERN_NBA_ADJUSTMENTS['totals'],
        'home_win_prob': 58.0,
        'away_win_prob': 42.0
    }

def main():
    # Header
    st.markdown('<div class="main-header">🏀 NBA Betting Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enhanced with Best Bets Dashboard & Date Selector</div>', unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    if not models:
        st.error("⚠️ Models not loaded")
        return
    
    # Sidebar - Date Selector
    st.sidebar.title("📅 Date Selection")
    
    date_option = st.sidebar.radio(
        "Select Date",
        ["Today", "Tomorrow", "Custom Date"]
    )
    
    if date_option == "Today":
        selected_date = datetime.now()
    elif date_option == "Tomorrow":
        selected_date = datetime.now() + timedelta(days=1)
    else:
        selected_date = st.sidebar.date_input(
            "Pick a date",
            datetime.now(),
            min_value=datetime.now() - timedelta(days=30),
            max_value=datetime.now() + timedelta(days=14)
        )
        selected_date = datetime.combine(selected_date, datetime.min.time())
    
    st.sidebar.markdown(f"**Selected: {selected_date.strftime('%b %d, %Y')}**")
    
    # Navigation
    st.sidebar.title("⚙️ Settings")
    page = st.sidebar.radio(
        "Navigate",
        ["🔥 Best Bets", "📅 All Games", "🎯 Custom Prediction", "📊 Performance"]
    )
    
    # Show selected page
    if page == "🔥 Best Bets":
        show_best_bets_dashboard(models, selected_date)
    elif page == "📅 All Games":
        st.header(f"📅 All Games - {selected_date.strftime('%A, %B %d, %Y')}")
        st.info("This will show all games for the selected date with predictions")
    elif page == "🎯 Custom Prediction":
        st.header("🎯 Custom Prediction")
        st.info("Select any two teams to get predictions")
    else:
        st.header("📊 Model Performance")
        st.info("View model accuracy and statistics")

if __name__ == "__main__":
    main()

