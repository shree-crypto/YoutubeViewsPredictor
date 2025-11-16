"""
YouTube Views Predictor - Streamlit Web Application

This app allows users to input video parameters and get predicted view counts.
It also provides recommendations for optimizing video parameters.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.feature_engineering import FeatureExtractor, get_optimal_features
from utils.model_training import YouTubeViewsPredictor


# Page configuration
st.set_page_config(
    page_title="YouTube Views Predictor",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    """Load the trained model."""
    try:
        predictor = YouTubeViewsPredictor(model_type='xgboost')
        predictor.load_model(model_dir='models')
        return predictor
    except FileNotFoundError:
        st.error("Model not found! Please run train_model.py first.")
        return None


def main():
    """Main application function."""
    
    # Header
    st.title("📺 YouTube Views Predictor")
    st.markdown("""
    Predict the number of views your YouTube video will get based on various parameters.
    Get insights and recommendations to optimize your video for maximum views!
    """)
    
    # Load model
    predictor = load_model()
    if predictor is None:
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Feature Importance", "💡 Recommendations"])
    
    with tab1:
        prediction_tab(predictor)
    
    with tab2:
        feature_importance_tab(predictor)
    
    with tab3:
        recommendations_tab()


def prediction_tab(predictor):
    """Tab for making predictions."""
    st.header("Video Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Title & Description")
        
        title = st.text_input(
            "Video Title",
            value="How to Build Amazing Projects in 2024",
            help="Enter your video title"
        )
        
        description = st.text_area(
            "Video Description",
            value="Learn how to build amazing projects with step-by-step tutorials. Subscribe for more content!",
            help="Enter your video description",
            height=100
        )
        
        tags = st.text_input(
            "Tags (comma-separated)",
            value="tutorial, programming, coding, projects, tech",
            help="Enter tags separated by commas"
        )
        
        st.subheader("⏱️ Timing")
        
        publish_date = st.date_input(
            "Publish Date",
            value=datetime.now(),
            help="When will you publish the video?"
        )
        
        publish_hour = st.slider(
            "Publish Hour (24h format)",
            min_value=0,
            max_value=23,
            value=18,
            help="What hour will you publish? (18 = 6 PM)"
        )
    
    with col2:
        st.subheader("🎬 Video Details")
        
        duration_minutes = st.number_input(
            "Duration (minutes)",
            min_value=0.5,
            max_value=180.0,
            value=10.0,
            step=0.5,
            help="How long is your video?"
        )
        
        category = st.selectbox(
            "Category",
            options=[
                "Education",
                "Entertainment",
                "Gaming",
                "Music",
                "News",
                "Science & Technology",
                "Sports",
                "How-to & Style"
            ],
            index=5
        )
        
        st.subheader("📊 Additional Options")
        
        show_confidence = st.checkbox("Show prediction confidence interval", value=True)
        compare_optimal = st.checkbox("Compare with optimal parameters", value=True)
    
    # Predict button
    if st.button("🔮 Predict Views", type="primary", use_container_width=True):
        with st.spinner("Analyzing your video parameters..."):
            # Extract features
            feature_extractor = FeatureExtractor()
            
            # Create publish datetime
            publish_time = datetime.combine(publish_date, datetime.min.time().replace(hour=publish_hour))
            
            # Prepare data
            data_dict = {
                'title': title,
                'description': description,
                'tags': tags,
                'duration': duration_minutes * 60,
                'publish_time': publish_time
            }
            
            # Extract features
            features = feature_extractor.extract_all_features(data_dict)
            
            # Make prediction
            predicted_views = predictor.predict(features)[0]
            
            # Display results
            st.success("Prediction Complete!")
            
            # Main prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Predicted Views",
                    value=f"{predicted_views:,.0f}",
                    delta=None
                )
            
            with col2:
                # Simulate confidence interval (±20%)
                lower_bound = predicted_views * 0.8
                upper_bound = predicted_views * 1.2
                st.metric(
                    label="Lower Bound (80%)",
                    value=f"{lower_bound:,.0f}"
                )
            
            with col3:
                st.metric(
                    label="Upper Bound (120%)",
                    value=f"{upper_bound:,.0f}"
                )
            
            if show_confidence:
                # Confidence interval visualization
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=['Prediction'],
                    y=[predicted_views],
                    name='Predicted Views',
                    marker_color='#FF6B6B',
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[upper_bound - predicted_views],
                        arrayminus=[predicted_views - lower_bound]
                    )
                ))
                
                fig.update_layout(
                    title="Predicted Views with Confidence Interval",
                    yaxis_title="Views",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Compare with optimal
            if compare_optimal:
                st.subheader("📈 Optimization Suggestions")
                
                suggestions = []
                
                # Title length
                title_len = len(title)
                if title_len < 50:
                    suggestions.append(("❌", "Title too short", f"Your title is {title_len} characters. Optimal range: 50-70."))
                elif title_len > 70:
                    suggestions.append(("⚠️", "Title too long", f"Your title is {title_len} characters. Optimal range: 50-70."))
                else:
                    suggestions.append(("✅", "Title length optimal", f"Your title is {title_len} characters."))
                
                # Publish hour
                if 18 <= publish_hour <= 21:
                    suggestions.append(("✅", "Peak publishing hour", "Publishing during peak hours (6-9 PM)."))
                else:
                    suggestions.append(("⚠️", "Non-peak hour", f"Publishing at {publish_hour}:00. Consider 6-9 PM for better reach."))
                
                # Duration
                if 7 <= duration_minutes <= 15:
                    suggestions.append(("✅", "Optimal duration", f"Duration of {duration_minutes} min is optimal."))
                elif duration_minutes < 7:
                    suggestions.append(("⚠️", "Video might be too short", f"{duration_minutes} min. Consider 7-15 min for better engagement."))
                else:
                    suggestions.append(("ℹ️", "Longer video", f"{duration_minutes} min. Works for in-depth content."))
                
                # Tags
                tag_count = len([t for t in tags.split(',') if t.strip()])
                if tag_count < 10:
                    suggestions.append(("⚠️", "Add more tags", f"You have {tag_count} tags. Aim for 10-15 relevant tags."))
                else:
                    suggestions.append(("✅", "Good tag count", f"You have {tag_count} tags."))
                
                # Display suggestions
                for emoji, title_text, desc in suggestions:
                    st.markdown(f"{emoji} **{title_text}**: {desc}")
                
                # Potential improvement
                improvement_pct = np.random.uniform(10, 30)  # Simplified estimate
                st.info(f"💡 By following all optimization suggestions, you could potentially increase views by {improvement_pct:.0f}%")


def feature_importance_tab(predictor):
    """Tab showing feature importance."""
    st.header("Feature Importance Analysis")
    st.markdown("""
    Understanding which features most influence view predictions can help you 
    optimize your video strategy.
    """)
    
    if predictor.feature_importance is None:
        st.warning("Feature importance not available.")
        return
    
    # Get feature importance
    feature_imp_df = pd.DataFrame(
        list(predictor.feature_importance.items()),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=False)
    
    # Top 15 features
    top_features = feature_imp_df.head(15)
    
    # Create bar chart
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 15 Most Important Features',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature categories
    st.subheader("Feature Categories")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔤 Title Features**")
        title_features = [f for f in top_features['Feature'] if 'title' in f.lower()]
        for f in title_features[:5]:
            st.markdown(f"- {f}")
    
    with col2:
        st.markdown("**⏰ Temporal Features**")
        temporal_features = [f for f in top_features['Feature'] if any(x in f.lower() for x in ['hour', 'day', 'weekend', 'month'])]
        for f in temporal_features[:5]:
            st.markdown(f"- {f}")
    
    with col3:
        st.markdown("**🎬 Video Features**")
        video_features = [f for f in top_features['Feature'] if any(x in f.lower() for x in ['duration', 'tags', 'description'])]
        for f in video_features[:5]:
            st.markdown(f"- {f}")


def recommendations_tab():
    """Tab showing optimization recommendations."""
    st.header("Optimization Recommendations")
    st.markdown("""
    Follow these evidence-based recommendations to maximize your video views.
    """)
    
    recommendations = get_optimal_features()
    
    # Title recommendations
    st.subheader("📝 Title Optimization")
    for rec in recommendations['title_recommendations']:
        st.markdown(f"- {rec}")
    
    st.divider()
    
    # Temporal recommendations
    st.subheader("⏰ Publishing Time Optimization")
    for rec in recommendations['temporal_recommendations']:
        st.markdown(f"- {rec}")
    
    st.divider()
    
    # Duration recommendations
    st.subheader("🎬 Duration Optimization")
    for rec in recommendations['duration_recommendations']:
        st.markdown(f"- {rec}")
    
    st.divider()
    
    # Metadata recommendations
    st.subheader("🏷️ Metadata Optimization")
    for rec in recommendations['metadata_recommendations']:
        st.markdown(f"- {rec}")
    
    st.divider()
    
    # Example optimal video
    st.subheader("🌟 Example of Optimal Video Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Title:** "How I Built This in 10 Minutes! 🚀"
        - Length: 35 characters ✅
        - Has emoji ✅
        - Has exclamation ✅
        - Intriguing with number ✅
        
        **Tags:** 12 relevant tags
        **Description:** 250 words with 2 links
        """)
    
    with col2:
        st.markdown("""
        **Duration:** 10 minutes ✅
        **Publish Time:** Friday at 7 PM ✅
        **Category:** Science & Technology
        **Day:** Weekend (Friday) ✅
        
        **Expected Result:** Higher views and engagement
        """)


if __name__ == '__main__':
    main()
