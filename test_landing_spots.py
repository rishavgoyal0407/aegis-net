"""
Automated test of landing spot detection system
"""
from agent import LandingSpotDetector
import cv2
import numpy as np

def test_image_analysis():
    """Test landing spot detection on drone image"""
    print("=" * 60)
    print("TESTING LANDING SPOT DETECTION")
    print("=" * 60)
    
    # Create detector
    detector = LandingSpotDetector(grid_size=20)
    
    # Analyze image
    print("\nAnalyzing drone image...")
    image_path = "photos/drone img1.jpg"
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read {image_path}")
        return
    
    # Perform analysis
    safety_map, heatmap_overlay = detector.analyze_frame(frame)
    
    # Find best landing spots
    best_spot = np.unravel_index(np.argmax(safety_map), safety_map.shape)
    max_score = safety_map[best_spot]
    avg_score = np.mean(safety_map)
    min_score = np.min(safety_map)
    
    # Print results
    print(f"\n{'='*60}")
    print("ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Grid Size: {detector.grid_size}x{detector.grid_size}")
    print(f"\nSafety Score Statistics:")
    print(f"  - Maximum (Best):  {max_score:.2f}/100")
    print(f"  - Average:         {avg_score:.2f}/100")
    print(f"  - Minimum (Worst): {min_score:.2f}/100")
    print(f"\nBest Landing Spot Location:")
    print(f"  - Grid Position: ({best_spot[1]}, {best_spot[0]})")
    print(f"  - Safety Score: {max_score:.2f}/100")
    
    # Find top 5 safest spots
    print(f"\nTop 5 Safest Landing Spots:")
    flat_indices = np.argsort(safety_map.flatten())[::-1][:5]
    for i, idx in enumerate(flat_indices, 1):
        spot = np.unravel_index(idx, safety_map.shape)
        score = safety_map[spot]
        print(f"  {i}. Grid ({spot[1]:2d}, {spot[0]:2d}) - Score: {score:.2f}/100")
    
    # Save visualizations
    cv2.imwrite("landing_heatmap_overlay.jpg", heatmap_overlay)
    print(f"\n{'='*60}")
    print("Saved: landing_heatmap_overlay.jpg")
    
    # Create and save matplotlib heatmap
    detector.plot_safety_heatmap(safety_map, best_spot)
    print("Saved: safety_heatmap.png")
    print(f"{'='*60}")
    
    # Display windows
    print("\nDisplaying results... (Press any key to close)")
    
    # Resize for better viewing
    display_width = 1200
    aspect_ratio = frame.shape[0] / frame.shape[1]
    display_height = int(display_width * aspect_ratio)
    
    frame_resized = cv2.resize(frame, (display_width, display_height))
    heatmap_resized = cv2.resize(heatmap_overlay, (display_width, display_height))
    
    # Stack images side by side
    comparison = np.hstack([frame_resized, heatmap_resized])
    
    cv2.imshow("Landing Spot Analysis: Original | Heatmap", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_image_analysis()
