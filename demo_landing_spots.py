"""
Quick demo script for landing spot detection
"""
import cv2
from agent import LandingSpotDetector

# Create detector with 15x15 grid
detector = LandingSpotDetector(grid_size=15)

# Example 1: Analyze a single image
print("Example 1: Analyzing single image...")
print("-" * 50)
detector.process_image("photos/drone img1.jpg", show_plot=True)

# Example 2: Analyze video (processes every 20th frame for speed)
# Uncomment to run:
# print("\nExample 2: Analyzing video...")
# print("-" * 50)
# detector.process_video("videos/Bluemlisalphutte Flyover.mp4", sample_rate=20)

# Example 3: Get raw safety scores
print("\nExample 3: Getting raw safety scores...")
print("-" * 50)
frame = cv2.imread("photos/drone img1.jpg")
safety_map, heatmap = detector.analyze_frame(frame)

# Find top 3 safest spots
import numpy as np
flat_indices = np.argsort(safety_map.flatten())[::-1][:3]
top_spots = [np.unravel_index(idx, safety_map.shape) for idx in flat_indices]

print("\nTop 3 Safest Landing Spots:")
for i, spot in enumerate(top_spots, 1):
    score = safety_map[spot]
    print(f"  {i}. Grid position ({spot[1]}, {spot[0]}) - Score: {score:.2f}/100")
