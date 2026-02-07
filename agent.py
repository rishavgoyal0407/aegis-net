import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class LandingSpotDetector:
    def __init__(self, grid_size=20):
        """
        Initialize the landing spot detector
        
        Args:
            grid_size: Number of grid cells in each dimension for analysis
        """
        self.grid_size = grid_size
        
    def analyze_flatness(self, region):
        """
        Analyze terrain flatness using edge detection
        Lower edge density = flatter surface = safer
        """
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (region.shape[0] * region.shape[1])
        
        # Convert to safety score (less edges = higher score)
        flatness_score = max(0, 100 - edge_density * 5)
        return flatness_score
    
    def analyze_texture(self, region):
        """
        Analyze surface texture uniformity
        Lower variance = smoother surface = safer
        """
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)
        
        # Convert to safety score (less variance = higher score)
        texture_score = max(0, 100 - variance / 10)
        return texture_score
    
    def analyze_gradient(self, region):
        """
        Analyze surface gradient/slope
        Lower gradient = less slope = safer
        """
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = np.mean(gradient_magnitude)
        
        # Convert to safety score (less gradient = higher score)
        gradient_score = max(0, 100 - avg_gradient / 2)
        return gradient_score
    
    def analyze_color_uniformity(self, region):
        """
        Analyze color uniformity
        Uniform colors often indicate flat surfaces (grass, concrete, etc.)
        """
        # Calculate standard deviation across color channels
        std_b = np.std(region[:, :, 0])
        std_g = np.std(region[:, :, 1])
        std_r = np.std(region[:, :, 2])
        
        avg_std = (std_b + std_g + std_r) / 3
        
        # Convert to safety score (less variation = higher score)
        color_score = max(0, 100 - avg_std / 3)
        return color_score
    
    def calculate_safety_score(self, region):
        """
        Calculate overall safety score by combining multiple factors
        
        Returns:
            Safety score from 0 (unsafe) to 100 (safe)
        """
        if region.size == 0:
            return 0
        
        # Calculate individual metrics
        flatness = self.analyze_flatness(region)
        texture = self.analyze_texture(region)
        gradient = self.analyze_gradient(region)
        color = self.analyze_color_uniformity(region)
        
        # Weighted average (you can adjust weights based on importance)
        weights = {
            'flatness': 0.35,
            'texture': 0.25,
            'gradient': 0.30,
            'color': 0.10
        }
        
        safety_score = (
            weights['flatness'] * flatness +
            weights['texture'] * texture +
            weights['gradient'] * gradient +
            weights['color'] * color
        )
        
        return safety_score
    
    def analyze_frame(self, frame):
        """
        Analyze a single frame and create safety heatmap
        
        Args:
            frame: Input image/frame from drone footage
            
        Returns:
            safety_map: 2D array of safety scores
            heatmap_overlay: Visual heatmap overlaid on original frame
        """
        height, width = frame.shape[:2]
        cell_height = height // self.grid_size
        cell_width = width // self.grid_size
        
        # Initialize safety map
        safety_map = np.zeros((self.grid_size, self.grid_size))
        
        # Analyze each grid cell
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Extract region
                y_start = i * cell_height
                y_end = min((i + 1) * cell_height, height)
                x_start = j * cell_width
                x_end = min((j + 1) * cell_width, width)
                
                region = frame[y_start:y_end, x_start:x_end]
                
                # Calculate safety score
                safety_map[i, j] = self.calculate_safety_score(region)
        
        # Create heatmap overlay
        heatmap_overlay = self.create_heatmap_overlay(frame, safety_map)
        
        return safety_map, heatmap_overlay
    
    def create_heatmap_overlay(self, frame, safety_map):
        """
        Create a visual heatmap overlay on the original frame
        Red = Unsafe, Green = Safe
        """
        height, width = frame.shape[:2]
        
        # Resize safety map to frame size
        heatmap_resized = cv2.resize(safety_map.astype(np.float32), 
                                      (width, height), 
                                      interpolation=cv2.INTER_LINEAR)
        
        # Normalize to 0-255
        heatmap_normalized = cv2.normalize(heatmap_resized, None, 0, 255, 
                                           cv2.NORM_MINMAX).astype(np.uint8)
        
        # Create color map (Red -> Yellow -> Green)
        # Red (unsafe) = 0, Green (safe) = 255
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # For better visualization: invert so green is safe and red is unsafe
        heatmap_colored = cv2.applyColorMap(255 - heatmap_normalized, cv2.COLORMAP_JET)
        
        # Blend with original frame
        overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
        
        return overlay
    
    def process_video(self, video_path, output_path=None, sample_rate=30):
        """
        Process drone video and create heatmap visualization
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            sample_rate: Process every Nth frame (default: 30)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        print("Processing video...")
        print("Press 'q' to quit, 's' to save current frame analysis")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Process every Nth frame for efficiency
            if frame_count % sample_rate == 0:
                # Analyze frame
                safety_map, heatmap_overlay = self.analyze_frame(frame)
                
                # Add text showing best landing spot
                best_spot = np.unravel_index(np.argmax(safety_map), safety_map.shape)
                max_score = safety_map[best_spot]
                
                text = f"Best Spot: ({best_spot[1]}, {best_spot[0]}) | Score: {max_score:.1f}/100"
                cv2.putText(heatmap_overlay, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add text to original frame too
                frame_with_text = frame.copy()
                cv2.putText(frame_with_text, "Original Drone Footage", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display in separate windows
                cv2.imshow("Original Drone Footage", frame_with_text)
                cv2.imshow("Landing Spot Heatmap", heatmap_overlay)
                
                # Write to output if specified
                if output_path:
                    out.write(heatmap_overlay)
                
                # Wait for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current analysis
                    cv2.imwrite("landing_analysis.jpg", heatmap_overlay)
                    cv2.imwrite("original_frame.jpg", frame)
                    self.plot_safety_heatmap(safety_map)
                    print(f"Saved frame analysis (Score: {max_score:.1f})")
            else:
                # Just display original frame without processing
                cv2.imshow("Original Drone Footage", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Processed {frame_count} frames")
    
    def process_image(self, image_path, show_plot=True):
        """
        Process a single image and create heatmap
        
        Args:
            image_path: Path to input image
            show_plot: Whether to display matplotlib plot
        """
        # Read image
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        # Analyze frame
        safety_map, heatmap_overlay = self.analyze_frame(frame)
        
        # Find best landing spot
        best_spot = np.unravel_index(np.argmax(safety_map), safety_map.shape)
        max_score = safety_map[best_spot]
        
        print(f"\nAnalysis Results:")
        print(f"Best Landing Spot: Grid position ({best_spot[1]}, {best_spot[0]})")
        print(f"Safety Score: {max_score:.2f}/100")
        print(f"Average Safety Score: {np.mean(safety_map):.2f}/100")
        
        # Display results
        cv2.imshow("Original Image", frame)
        cv2.imshow("Landing Spot Heatmap", heatmap_overlay)
        
        if show_plot:
            self.plot_safety_heatmap(safety_map, best_spot)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return safety_map, heatmap_overlay
    
    def plot_safety_heatmap(self, safety_map, best_spot=None):
        """
        Create a detailed matplotlib heatmap visualization
        """
        # Create custom colormap (Red -> Yellow -> Green)
        colors = ['red', 'yellow', 'green']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('safety', colors, N=n_bins)
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        im = plt.imshow(safety_map, cmap=cmap, interpolation='bilinear', 
                       vmin=0, vmax=100)
        
        # Add colorbar
        cbar = plt.colorbar(im, label='Safety Score')
        cbar.set_label('Safety Score (0=Unsafe, 100=Safe)', rotation=270, labelpad=20)
        
        # Mark best spot if provided
        if best_spot is not None:
            plt.plot(best_spot[1], best_spot[0], 'w*', markersize=20, 
                    markeredgecolor='black', markeredgewidth=2, 
                    label='Best Landing Spot')
            plt.legend()
        
        plt.title('Landing Spot Safety Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Grid Column')
        plt.ylabel('Grid Row')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('safety_heatmap.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Heatmap saved as 'safety_heatmap.png'")


def main():
    """
    Main function to run landing spot detection
    """
    # Create detector instance
    detector = LandingSpotDetector(grid_size=20)
    
    print("=" * 60)
    print("DRONE LANDING SPOT DETECTION SYSTEM")
    print("=" * 60)
    print("\nOptions:")
    print("1. Analyze drone video")
    print("2. Analyze single image")
    print("3. Process video and save output")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        # Analyze video
        video_path = input("Enter video path (or press Enter for default): ").strip()
        if not video_path:
            video_path = "videos/Bluemlisalphutte Flyover.mp4"
        
        print("\nTwo windows will open:")
        print("  - Original Drone Footage (raw video)")
        print("  - Landing Spot Heatmap (safety analysis)")
        print("\nPress 'q' to quit, 's' to save current frame\n")
        
        detector.process_video(video_path, sample_rate=15)
        
    elif choice == "2":
        # Analyze image
        image_path = input("Enter image path (or press Enter for default): ").strip()
        if not image_path:
            image_path = "photos/drone img1.jpg"
        
        safety_map, heatmap = detector.process_image(image_path)
        
    elif choice == "3":
        # Process and save video
        video_path = input("Enter input video path: ").strip()
        output_path = input("Enter output video path: ").strip()
        
        if video_path and output_path:
            detector.process_video(video_path, output_path, sample_rate=10)
        else:
            print("Both paths are required!")
    
    else:
        print("Invalid option!")


if __name__ == "__main__":
    main()
