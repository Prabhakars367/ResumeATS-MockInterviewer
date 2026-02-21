import cv2
import time
import threading
import signal
import sys
from emotion_analyzer import EmotionAnalyzer
from datetime import datetime
import json

class FaceEmotionRecognitionSystem:
    def __init__(self, model_path='emotion_recognition_model.h5', log_file='emotion_analysis_log.json'):
        """
        Initialize the Face Emotion Recognition System
        """
        self.analyzer = EmotionAnalyzer(model_path)
        self.log_file = log_file
        self.running = False
        self.cap = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Setup signal handler for graceful shutdown only when running in main thread
        # Streamlit and some environments run code in worker threads where setting
        # signal handlers raises an exception. Wrap in try/except and only set
        # the handler if we're in the main thread.
        try:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, self.signal_handler)
        except Exception as e:
            # Non-fatal: just warn and continue. This prevents crashes under Streamlit.
            print(f"Warning: could not set signal handler: {e}")
    
    def signal_handler(self, sig, frame):
        """
        Handle Ctrl+C for graceful shutdown
        """
        print("\nShutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def start_camera(self, camera_index=0):
        """
        Initialize and start the camera
        """
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized successfully")
    
    def draw_analysis_overlay(self, frame, analysis_result):
        """
        Draw analysis results on the frame
        """
        if not analysis_result['face_detected']:
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        # Draw emotion information
        y_offset = 30
        dominant_emotion = analysis_result['dominant_emotion']
        confidence_score = analysis_result['confidence_score']
        nervousness_score = analysis_result['nervousness_score']
        
        # Dominant emotion
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        
        # Confidence score
        confidence_color = (0, 255, 0) if confidence_score > 0.6 else (0, 165, 255) if confidence_score > 0.3 else (0, 0, 255)
        cv2.putText(frame, f"Confidence: {confidence_score:.2f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, confidence_color, 2)
        y_offset += 30
        
        # Nervousness score
        nervousness_color = (0, 0, 255) if nervousness_score > 0.6 else (0, 165, 255) if nervousness_score > 0.3 else (0, 255, 0)
        cv2.putText(frame, f"Nervousness: {nervousness_score:.2f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, nervousness_color, 2)
        y_offset += 30
        
        # Draw emotion probabilities
        emotions = analysis_result['emotions']
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        
        cv2.putText(frame, "Top Emotions:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        for emotion, prob in top_emotions:
            cv2.putText(frame, f"{emotion}: {prob:.2f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # Draw facial features if available
        features = analysis_result['facial_features']
        if features:
            cv2.putText(frame, "Features:", (400, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            feature_y = 50
            cv2.putText(frame, f"Blink Rate: {features.get('blink_rate', 0):.1f}/min", 
                       (400, feature_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            feature_y += 20
            
            cv2.putText(frame, f"Head Movement: {features.get('head_movement_variance', 0):.1f}", 
                       (400, feature_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            feature_y += 20
            
            cv2.putText(frame, f"Symmetry: {features.get('facial_symmetry', 0):.2f}", 
                       (400, feature_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw performance info
        fps = self.frame_count / (time.time() - self.start_time) if time.time() - self.start_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw confidence and nervousness bars
        self.draw_progress_bar(frame, confidence_score, (400, frame.shape[0] - 60), "Confidence", (0, 255, 0))
        self.draw_progress_bar(frame, nervousness_score, (400, frame.shape[0] - 30), "Nervousness", (0, 0, 255))
        
        return frame
    
    def draw_progress_bar(self, frame, value, position, label, color):
        """
        Draw a progress bar for scores
        """
        bar_width = 200
        bar_height = 20
        
        # Background bar
        cv2.rectangle(frame, position, (position[0] + bar_width, position[1] + bar_height), 
                     (50, 50, 50), -1)
        
        # Progress bar
        progress_width = int(bar_width * value)
        cv2.rectangle(frame, position, (position[0] + progress_width, position[1] + bar_height), 
                     color, -1)
        
        # Label and value
        cv2.putText(frame, f"{label}: {value:.2f}", (position[0], position[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self, show_video=True):
        """
        Run the emotion recognition system
        """
        try:
            self.start_camera()
            self.running = True
            
            print("Starting Face Emotion Recognition System...")
            print("Press 'q' to quit, 's' to save current analysis")
            print(f"Logging data every {self.analyzer.log_interval} seconds to {self.log_file}")
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Process frame for emotion analysis
                analysis_result = self.analyzer.process_frame(frame)
                
                # Log the analysis
                self.analyzer.log_analysis(analysis_result, self.log_file)
                
                # Update frame count for FPS calculation
                self.frame_count += 1
                
                if show_video:
                    # Draw analysis overlay
                    annotated_frame = self.draw_analysis_overlay(frame, analysis_result)
                    
                    # Display the frame
                    cv2.imshow('Face Emotion Recognition System', annotated_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save current analysis summary
                        summary = self.analyzer.get_analysis_summary()
                        if summary:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            summary_file = f"emotion_summary_{timestamp}.json"
                            with open(summary_file, 'w') as f:
                                json.dump(summary, f, indent=2)
                            print(f"Analysis summary saved to {summary_file}")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """
        Stop the emotion recognition system
        """
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Force log any remaining data
        if self.analyzer.log_data:
            self.analyzer.log_analysis({}, self.log_file)
        
        print("System stopped successfully")

def main():
    """
    Main function to run the application
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Emotion Recognition System')
    parser.add_argument('--model', default='emotion_recognition_model.h5', 
                       help='Path to the emotion recognition model')
    parser.add_argument('--log-file', default='emotion_analysis_log.json', 
                       help='Path to the log file')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index to use')
    parser.add_argument('--no-video', action='store_true', 
                       help='Run without video display (headless mode)')
    
    args = parser.parse_args()
    
    # Create and run the system
    system = FaceEmotionRecognitionSystem(args.model, args.log_file)
    
    try:
        system.run(show_video=not args.no_video)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
