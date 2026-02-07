"""
Video Analysis Example - Process drone video with landing spot detection
"""
from agent import LandingSpotDetector

def analyze_video_with_best_settings():
    """
    Demonstrates video analysis with optimized settings
    """
    print("=" * 70)
    print("VIDEO LANDING SPOT ANALYSIS")
    print("=" * 70)
    
    # Create detector with optimal grid size
    detector = LandingSpotDetector(grid_size=15)
    
    # List available videos
    videos = [
        "videos/Bluemlisalphutte Flyover.mp4",
        "videos/Creux du Van Flight.mp4",
        "videos/Stockflue Flyaround.mp4",
    ]
    
    print("\nAvailable videos:")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video.split('/')[-1]}")
    
    print("\nAnalyzing video with landing spot detection...")
    print("Two windows will appear:")
    print("  1. Original Drone Footage - Raw video feed")
    print("  2. Landing Spot Heatmap - Safety analysis overlay")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame analysis")
    print("-" * 70)
    
    # Process video (every 20th frame for smooth visualization)
    # Green areas = safe landing spots
    # Red areas = unsafe landing spots
    detector.process_video(videos[0], sample_rate=20)
    
    print("\nVideo analysis complete!")

if __name__ == "__main__":
    analyze_video_with_best_settings()
