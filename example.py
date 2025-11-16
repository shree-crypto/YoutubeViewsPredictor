#!/usr/bin/env python3
"""
Example Script - YouTube Views Predictor

This script demonstrates basic usage of the predictor with example videos.
Run this after training the model with: python train_model.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_training import YouTubeViewsPredictor
from utils.feature_engineering import FeatureExtractor


def print_separator():
    """Print a visual separator."""
    print("\n" + "=" * 70)


def predict_video(predictor, extractor, video_data, video_name):
    """Make a prediction for a video and display results."""
    print(f"\n🎥 {video_name}")
    print("-" * 70)
    print(f"Title: {video_data['title']}")
    print(f"Duration: {video_data['duration'] / 60:.0f} minutes")
    print(f"Tags: {video_data['tags']}")
    print(f"Publish Time: {video_data['publish_time']}")
    
    # Extract features and predict
    features = extractor.extract_all_features(video_data)
    predicted_views = predictor.predict(features)[0]
    
    print(f"\n📊 Predicted Views: {predicted_views:,.0f}")
    print(f"💰 Estimated Revenue (CPM $2): ${predicted_views * 0.002:.2f}")
    
    return predicted_views


def main():
    """Run example predictions."""
    print_separator()
    print("🎬 YouTube Views Predictor - Example Predictions")
    print_separator()
    
    # Load the trained model
    print("\n⏳ Loading trained model...")
    try:
        predictor = YouTubeViewsPredictor(model_type='xgboost')
        predictor.load_model('models')
        print("✓ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Error: Model not found!")
        print("Please run 'python train_model.py' first to train the model.")
        sys.exit(1)
    
    # Create feature extractor
    extractor = FeatureExtractor()
    
    print_separator()
    print("📝 Example Videos")
    print_separator()
    
    # Example 1: Python Tutorial (Non-optimized)
    video1 = {
        'title': 'Python Tutorial',
        'duration': 600,  # 10 minutes
        'tags': 'python,tutorial',
        'publish_time': '2024-01-15 14:00:00',  # 2 PM Monday
        'description': 'Learn Python basics in this tutorial.'
    }
    views1 = predict_video(predictor, extractor, video1, "Example 1: Basic Tutorial (Non-optimized)")
    
    # Example 2: Python Tutorial (Optimized)
    video2 = {
        'title': 'Learn Python in 10 Minutes! Complete Beginner Guide 2024',
        'duration': 600,
        'tags': 'python,tutorial,programming,beginners,coding,2024,learn,howto,python3,development',
        'publish_time': '2024-01-19 19:00:00',  # 7 PM Friday
        'description': '''
Learn Python programming from scratch in just 10 minutes! This complete beginner-friendly 
tutorial covers variables, loops, functions, and more. Perfect for anyone starting their 
coding journey in 2024.

🎯 What You'll Learn:
• Python basics and syntax
• Variables and data types
• Control flow and loops
• Functions and modules

📚 Resources: https://python.org
⏱️ Timestamps:
0:00 - Introduction
2:00 - Variables
4:00 - Loops
7:00 - Functions
        '''
    }
    views2 = predict_video(predictor, extractor, video2, "Example 2: Optimized Tutorial")
    
    # Example 3: Tech Review
    video3 = {
        'title': 'Is This the BEST Budget Laptop Under $500? Honest Review 2024',
        'duration': 600,  # 10 minutes
        'tags': 'laptop,review,tech,budget,2024,technology,unboxing,best,affordable,gadgets,computer',
        'publish_time': '2024-02-10 18:30:00',  # 6:30 PM Saturday
        'description': '''
An honest, in-depth review of the XYZ laptop - is it really the best budget option 
under $500 in 2024? I tested everything: performance, build quality, battery life, and more.

💻 Specs Covered:
• Processor & RAM
• Graphics performance
• Display quality
• Battery life

✅ Pros and ❌ Cons discussed in detail.

⏱️ Timestamps:
0:00 - Unboxing
1:30 - Design & Build
3:00 - Performance Tests
5:00 - Battery Life
7:00 - Pros & Cons
8:30 - Final Verdict
        '''
    }
    views3 = predict_video(predictor, extractor, video3, "Example 3: Tech Review")
    
    # Example 4: Gaming Content
    video4 = {
        'title': '10 SECRET Tips to Dominate in Gaming! 🎮',
        'duration': 720,  # 12 minutes
        'tags': 'gaming,game,tips,tricks,tutorial,guide,howto,pro,strategy,walkthrough,gameplay',
        'publish_time': '2024-03-15 20:00:00',  # 8 PM Friday
        'description': '''
Master your favorite games with these 10 SECRET tips that pro players use! 
Boost your rank and dominate the competition with these advanced strategies.

🎮 Tips Covered:
1. Advanced movement techniques
2. Best loadout combinations
3. Map control strategies
4. Resource management
5. Positioning tips

⏱️ Timestamps included for each tip.
        '''
    }
    views4 = predict_video(predictor, extractor, video4, "Example 4: Gaming Content")
    
    # Example 5: Educational Content
    video5 = {
        'title': 'How Does the Internet Actually Work? Explained Simply! 🌐',
        'duration': 720,  # 12 minutes
        'tags': 'education,science,technology,internet,explained,learning,tutorial,networking,computer',
        'publish_time': '2024-05-10 19:30:00',  # 7:30 PM Friday
        'description': '''
The internet is everywhere, but how does it ACTUALLY work? This video breaks down the 
complex technology behind the internet into simple, easy-to-understand concepts!

🌐 Topics Covered:
• What is the internet?
• How data travels
• IP addresses explained
• DNS and domain names
• Network protocols

⏱️ Full chapter markers included!
        '''
    }
    views5 = predict_video(predictor, extractor, video5, "Example 5: Educational Content")
    
    # Summary
    print_separator()
    print("📊 Summary & Insights")
    print_separator()
    
    improvement = ((views2 - views1) / views1) * 100
    total_views = views1 + views2 + views3 + views4 + views5
    avg_views = total_views / 5
    
    print(f"\n🔍 Key Insights:")
    print(f"• Basic vs Optimized Tutorial: +{improvement:.1f}% improvement")
    print(f"• Average predicted views: {avg_views:,.0f}")
    print(f"• Total potential views: {total_views:,.0f}")
    print(f"• Best performing: Gaming Content ({views4:,.0f} views)")
    
    print("\n💡 Optimization Tips Applied:")
    print("✓ Peak hour publishing (6-9 PM)")
    print("✓ Weekend uploads (Friday-Sunday)")
    print("✓ Question marks in titles")
    print("✓ Numbers in titles (e.g., '10 Tips')")
    print("✓ Year for freshness (2024)")
    print("✓ Comprehensive descriptions with timestamps")
    print("✓ 10-15 relevant tags")
    print("✓ Emojis for visual appeal")
    
    print("\n🎯 Top Features That Drive Views:")
    top_features = predictor.get_top_features(n=5)
    for i, (feature, importance) in enumerate(top_features, 1):
        print(f"{i}. {feature}: {importance:.4f}")
    
    print_separator()
    print("\n🚀 Next Steps:")
    print("1. Try the web interface: streamlit run app.py")
    print("2. Experiment with your own video ideas")
    print("3. Check EXAMPLES.md for more scenarios")
    print("4. Read GETTING_STARTED.md for detailed guide")
    print_separator()
    print("\n✨ Happy predicting!")


if __name__ == '__main__':
    main()
