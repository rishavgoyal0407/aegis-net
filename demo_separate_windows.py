"""
Quick demonstration of real-time heatmap in separate windows
This script shows a brief preview of the two-window display
"""
from agent import LandingSpotDetector
import cv2

def quick_demo():
    """
    Quick demo showing the separate window feature
    """
    print("=" * 70)
    print("REAL-TIME LANDING SPOT DETECTION - SEPARATE WINDOWS DEMO")
    print("=" * 70)
    
    # Create detector
    detector = LandingSpotDetector(grid_size=15)
    
    # Load sample image for demonstration
    frame = cv2.imread("photos/drone img1.jpg")
    
    if frame is None:
        print("Error: Could not load sample image")
        return
    
    # Analyze frame
    print("\nAnalyzing frame and creating heatmap...")
    safety_map, heatmap_overlay = detector.analyze_frame(frame)
    
    # Add labels
    frame_labeled = frame.copy()
    cv2.putText(frame_labeled, "Original Drone Footage", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    best_spot = detector.grid_size - 1
    import numpy as np
    best_spot = np.unravel_index(np.argmax(safety_map), safety_map.shape)
    max_score = safety_map[best_spot]
    
    text = f"Landing Spot Heatmap - Best Score: {max_score:.1f}/100"
    cv2.putText(heatmap_overlay, text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    print("\nTwo separate windows will now appear:")
    print("  1. Original Drone Footage - Raw image")
    print("  2. Landing Spot Heatmap - Color-coded safety zones")
    print("\n  GREEN = Safe landing spots")
    print("  RED = Unsafe landing spots")
    print("\nPress any key to close windows...")
    
    # Display in separate windows
    cv2.imshow("Original Drone Footage", frame_labeled)
    cv2.imshow("Landing Spot Heatmap", heatmap_overlay)
    
    # Position windows side by side (if possible)
    cv2.moveWindow("Original Drone Footage", 50, 50)
    cv2.moveWindow("Landing Spot Heatmap", frame.shape[1] + 70, 50)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nDemo complete! For video analysis, run: python analyze_video.py")

if __name__ == "__main__":
    quick_demo()
