import pandas as pd
import random
import json
import threading
import time
import cv2
from datetime import datetime
from voice_rec import voice_rec
from face_emotion_system import FaceEmotionRecognitionSystem
from eye_tracking import EyeTracker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords')

class AIInterviewer:
    def __init__(self, csv_path='final_csv.csv', 
                 emotion_model_path='emotion_recognition_model.h5',
                 log_file='interview_log.json'):
        """
        Initialize the AI Interviewer with voice and emotion recognition
        """
        import os
        # Get the directory where the script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct absolute paths for all resources
        self.csv_path = os.path.join(current_dir, 'final_csv.csv')
        self.log_file = os.path.join(current_dir, log_file)
        self.questions_df = pd.read_csv(self.csv_path)
        
#         # Initialize voice recognition
#         self.voice_recognizer = None
        
#         # Initialize emotion recognition system
#         self.emotion_system = FaceEmotionRecognitionSystem(
#             model_path=emotion_model_path,
#             log_file='emotion_analysis_log.json'
#         )
        
#         # Interview data
#         self.current_interview = {
#             'timestamp': datetime.now().isoformat(),
#             'questions_asked': [],
#             'responses': [],
#             'emotion_analysis': [],
#             'similarity_scores': [],
#             'overall_performance': {}
#         }
        
#         # Text preprocessing
#         self.stop_words = set(stopwords.words('english'))
        
#     def preprocess_text(self, text):
#         """
#         Preprocess text for similarity comparison
#         """
#         # Convert to lowercase
#         text = text.lower()
        
#         # Remove special characters and numbers
#         text = re.sub(r'[^a-zA-Z\s]', '', text)
        
#         # Tokenize
#         words = word_tokenize(text)
        
#         # Remove stopwords
#         words = [word for word in words if word not in self.stop_words]
        
#         return ' '.join(words)
    
#     def calculate_similarity(self, user_answer, expected_answer):
#         """
#         Calculate similarity between user answer and expected answer using TF-IDF and cosine similarity
#         """
#         # Preprocess both texts
#         user_processed = self.preprocess_text(user_answer)
#         expected_processed = self.preprocess_text(expected_answer)
        
#         # Create TF-IDF vectors
#         vectorizer = TfidfVectorizer()
#         tfidf_matrix = vectorizer.fit_transform([user_processed, expected_processed])
        
#         # Calculate cosine similarity
#         similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
#         return float(similarity_score)
    
#     def get_random_questions(self, num_questions=5, job_role=None):
#         """
#         Get random questions from the CSV file
#         """
#         if job_role:
#             filtered_df = self.questions_df[self.questions_df['Job Role'] == job_role]
#             if len(filtered_df) == 0:
#                 print(f"No questions found for job role: {job_role}")
#                 filtered_df = self.questions_df
#         else:
#             filtered_df = self.questions_df
        
#         # Sample random questions
#         if len(filtered_df) < num_questions:
#             num_questions = len(filtered_df)
            
#         selected_questions = filtered_df.sample(n=num_questions)
#         return selected_questions.to_dict('records')
    
#     def start_emotion_monitoring(self):
#         """
#         Start emotion monitoring in a separate thread
#         """
#         def run_emotion_system():
#             try:
#                 self.emotion_system.run(show_video=True)
#             except Exception as e:
#                 print(f"Error in emotion monitoring: {e}")
        
#         emotion_thread = threading.Thread(target=run_emotion_system, daemon=True)
#         emotion_thread.start()
#         return emotion_thread
    
#     def get_emotion_summary(self):
#         """
#         Get emotion analysis summary for the current period
#         """
#         try:
#             return self.emotion_system.analyzer.get_analysis_summary()
#         except Exception as e:
#             print(f"Error getting emotion summary: {e}")
#             return None
    
#     def conduct_interview(self, num_questions=5, job_role=None):
#         """
#         Conduct the complete interview process
#         """
#         print("=" * 50)
#         print("AI INTERVIEWER - TECHNICAL ROUND")
#         print("=" * 50)
        
#         # Get available job roles
#         available_roles = self.questions_df['Job Role'].unique().tolist()
#         print(f"Available job roles: {', '.join(available_roles)}")
        
#         if not job_role:
#             print("\nSelect a job role or press Enter for random questions:")
#             selected_role = input().strip()
#             job_role = selected_role if selected_role in available_roles else None
        
#         # Get random questions
#         questions = self.get_random_questions(num_questions, job_role)
        
#         if not questions:
#             print("No questions available!")
#             return
        
#         # Initialize voice recognition with the questions
#         question_texts = [q['Question'] for q in questions]
#         self.voice_recognizer = voice_rec(question_texts)
        
#         # Start emotion monitoring
#         print("\nStarting emotion monitoring...")
#         emotion_thread = self.start_emotion_monitoring()
#         time.sleep(2)  # Give camera time to initialize
        
#         # Conduct interview
#         print(f"\nStarting interview with {len(questions)} questions...")
#         print("The interview will begin in 3 seconds...")
#         time.sleep(3)
        
#         # Introduction
#         self.voice_recognizer.speak("Hello! Welcome to your technical interview.")
#         self.voice_recognizer.speak("I will ask you several technical questions.")
#         self.voice_recognizer.speak("Please answer each question to the best of your ability.")
#         self.voice_recognizer.speak("Let's begin!")
        
#         # Ask each question
#         for i, question_data in enumerate(questions, 1):
#             print(f"\n--- Question {i}/{len(questions)} ---")
#             question = question_data['Question']
#             expected_answer = question_data['Answer']
#             job_role_q = question_data['Job Role']
            
#             print(f"Question: {question}")
            
#             # Ask the question using voice
#             self.voice_recognizer.speak(f"Question {i}.")
#             user_answer = self.voice_recognizer.listen(question)
            
#             if user_answer:
#                 print(f"Your answer: {user_answer}")
                
#                 # Calculate similarity
#                 similarity_score = self.calculate_similarity(user_answer, expected_answer)
                
#                 # Get emotion analysis for this question period
#                 emotion_summary = self.get_emotion_summary()
                
#                 # Store question data
#                 question_result = {
#                     'question_number': i,
#                     'job_role': job_role_q,
#                     'question': question,
#                     'expected_answer': expected_answer,
#                     'user_answer': user_answer,
#                     'similarity_score': similarity_score,
#                     'emotion_analysis': emotion_summary,
#                     'timestamp': datetime.now().isoformat()
#                 }
                
#                 self.current_interview['questions_asked'].append(question_result)
                
#                 # Provide feedback
#                 if similarity_score >= 0.7:
#                     feedback = "Excellent answer!"
#                 elif similarity_score >= 0.5:
#                     feedback = "Good answer!"
#                 elif similarity_score >= 0.3:
#                     feedback = "Fair answer, could be improved."
#                 else:
#                     feedback = "Please review this topic further."
                
#                 print(f"Similarity Score: {similarity_score:.2f}")
#                 print(f"Feedback: {feedback}")
                
#                 # Brief pause between questions
#                 time.sleep(2)
#             else:
#                 print("No answer recorded.")
        
#         # End interview
#         self.voice_recognizer.speak("Thank you for completing the interview.")
#         self.voice_recognizer.speak("Your performance will be analyzed and results will be saved.")
        
#         # Stop emotion monitoring
#         self.emotion_system.stop()
        
#         # Calculate overall performance
#         self.calculate_overall_performance()
        
#         # Save results
#         self.save_interview_results()
        
#         print("\nInterview completed! Results saved to", self.log_file)
    
#     def calculate_overall_performance(self):
#         """
#         Calculate overall interview performance metrics
#         """
#         if not self.current_interview['questions_asked']:
#             return
        
#         # Calculate average similarity score
#         similarity_scores = [q['similarity_score'] for q in self.current_interview['questions_asked']]
#         avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
#         # Calculate performance grade
#         if avg_similarity >= 0.8:
#             grade = "Excellent"
#         elif avg_similarity >= 0.6:
#             grade = "Good"
#         elif avg_similarity >= 0.4:
#             grade = "Fair"
#         else:
#             grade = "Needs Improvement"
        
#         # Analyze emotions (if available)
#         emotion_scores = []
#         confidence_scores = []
#         nervousness_scores = []
        
#         for question in self.current_interview['questions_asked']:
#             if question['emotion_analysis']:
#                 emotion_data = question['emotion_analysis']
#                 if 'average_emotions' in emotion_data:
#                     # Extract confidence and nervousness indicators
#                     emotions = emotion_data['average_emotions']
#                     confidence = emotions.get('happy', 0) + emotions.get('neutral', 0)
#                     nervousness = emotions.get('fear', 0) + emotions.get('sad', 0)
                    
#                     confidence_scores.append(confidence)
#                     nervousness_scores.append(nervousness)
        
#         avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
#         avg_nervousness = sum(nervousness_scores) / len(nervousness_scores) if nervousness_scores else 0
        
#         self.current_interview['overall_performance'] = {
#             'total_questions': len(self.current_interview['questions_asked']),
#             'average_similarity_score': avg_similarity,
#             'performance_grade': grade,
#             'average_confidence': avg_confidence,
#             'average_nervousness': avg_nervousness,
#             'individual_scores': similarity_scores
#         }
    
#     def save_interview_results(self):
#         """
#         Save interview results to JSON file
#         """
#         try:
#             # Load existing interviews if file exists
#             existing_interviews = []
#             try:
#                 with open(self.log_file, 'r') as f:
#                     existing_interviews = json.load(f)
#             except FileNotFoundError:
#                 pass
            
#             # Add current interview
#             existing_interviews.append(self.current_interview)
            
#             # Save to file
#             with open(self.log_file, 'w') as f:
#                 json.dump(existing_interviews, f, indent=2)
            
#             print(f"Interview results saved to {self.log_file}")
            
#         except Exception as e:
#             print(f"Error saving interview results: {e}")
    
#     def display_performance_summary(self):
#         """
#         Display a summary of the interview performance
#         """
#         performance = self.current_interview['overall_performance']
        
#         print("\n" + "=" * 50)
#         print("INTERVIEW PERFORMANCE SUMMARY")
#         print("=" * 50)
#         print(f"Total Questions: {performance['total_questions']}")
#         print(f"Average Similarity Score: {performance['average_similarity_score']:.2f}")
#         print(f"Performance Grade: {performance['performance_grade']}")
#         print(f"Average Confidence: {performance['average_confidence']:.2f}")
#         print(f"Average Nervousness: {performance['average_nervousness']:.2f}")
#         print("\nDetailed Scores:")
        
#         for i, score in enumerate(performance['individual_scores'], 1):
#             print(f"  Question {i}: {score:.2f}")
        
#         print("=" * 50)

# def main():
#     """
#     Main function to run the AI Interviewer
#     """
#     try:
#         # Initialize the AI Interviewer
#         interviewer = AIInterviewer()
        
#         # Get user preferences
#         print("Welcome to the AI Technical Interviewer!")
        
#         try:
#             num_questions = int(input("How many questions would you like? (default: 5): ") or "5")
#         except ValueError:
#             num_questions = 5
        
#         # Conduct the interview
#         interviewer.conduct_interview(num_questions=num_questions)
        
#         # Display performance summary
#         interviewer.display_performance_summary()
        
#     except KeyboardInterrupt:
#         print("\nInterview interrupted by user.")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()

import pandas as pd
import random
import json
import threading
import time
from datetime import datetime
from voice_rec import voice_rec
from face_emotion_system import FaceEmotionRecognitionSystem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class AIInterviewer:
    def __init__(self, csv_path=r'C:\Users\Shrey\Documents\Semester\code\projects\MAJOR\emotion_rec\final_csv.csv', 
                 emotion_model_path='emotion_recognition_model.h5',
                 log_file='interview_log.json'):
        """
        Initialize the AI Interviewer with voice and emotion recognition
        """
        self.csv_path = r"C:\Users\Shrey\Documents\Semester\code\projects\MAJOR\emotion_rec\final_csv.csv"
        self.log_file = log_file
        self.questions_df = pd.read_csv(self.csv_path)
        
        # Initialize voice recognition
        self.voice_recognizer = None
        
        # Initialize emotion recognition system
        self.emotion_system = FaceEmotionRecognitionSystem(
            model_path=emotion_model_path,
            log_file='emotion_analysis_log.json'
        )
        
        # Initialize eye tracking system
        self.eye_tracker = EyeTracker(
            log_file='eye_tracking_log.json'
        )
        
        # Interview data
        self.current_interview = {
            'timestamp': datetime.now().isoformat(),
            'questions_asked': [],
            'responses': [],
            'emotion_analysis': [],
            'eye_tracking_analysis': [],
            'similarity_scores': [],
            'overall_performance': {}
        }
        
        # Text preprocessing
        self.stop_words = set(stopwords.words('english'))
        
    def speak_and_print(self, text):
        """
        Both speak and print the text
        """
        print(text)
        if self.voice_recognizer:
            self.voice_recognizer.speak(text)
        
    def preprocess_text(self, text):
        """
        Preprocess text for similarity comparison
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stopwords
        words = [word for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def calculate_similarity(self, user_answer, expected_answer):
        """
        Calculate similarity between user answer and expected answer using TF-IDF and cosine similarity
        """
        # Preprocess both texts
        user_processed = self.preprocess_text(user_answer)
        expected_processed = self.preprocess_text(expected_answer)
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([user_processed, expected_processed])
        
        # Calculate cosine similarity
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity_score)
    
    def get_random_questions(self, num_questions=5, job_role=None):
        """
        Get random questions from the CSV file
        """
        if job_role:
            filtered_df = self.questions_df[self.questions_df['Job Role'] == job_role]
            if len(filtered_df) == 0:
                self.speak_and_print(f"No questions found for job role: {job_role}")
                filtered_df = self.questions_df
        else:
            filtered_df = self.questions_df
        
        # Sample random questions
        if len(filtered_df) < num_questions:
            num_questions = len(filtered_df)
            
        selected_questions = filtered_df.sample(n=num_questions)
        return selected_questions.to_dict('records')
    
    def start_emotion_monitoring(self):
        """
        Start emotion monitoring in a separate thread
        """
        def run_combined_camera():
            """
            Combined loop that reads frames from the camera, runs emotion analysis and eye-tracking,
            overlays both analyses on the same frame and displays it.
            """
            try:
                # Initialize camera from the emotion system (shared capture)
                self.emotion_system.start_camera()
                cap = self.emotion_system.cap
                self.emotion_system.running = True

                print("Starting combined Emotion + Eye-Tracking camera...")
                print("Press 'q' to quit, 's' to save summaries")

                while self.emotion_system.running:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read frame from camera")
                        break

                    # Run emotion analysis
                    analysis_result = self.emotion_system.analyzer.process_frame(frame)

                    # Run eye-tracking analysis
                    eye_analysis = self.eye_tracker.analyze_frame(frame)

                    # Log emotion analysis periodically
                    try:
                        self.emotion_system.analyzer.log_analysis(analysis_result, self.emotion_system.log_file)
                    except Exception:
                        pass

                    # Draw emotion overlay
                    annotated_frame = self.emotion_system.draw_analysis_overlay(frame.copy(), analysis_result)

                    # Draw eye-tracking overlay (simple text/indicators)
                    if eye_analysis:
                        gy = eye_analysis.get('gaze_y', 0.0)
                        la = eye_analysis.get('looking_away', False)
                        le = eye_analysis.get('left_ear', 0.0)
                        re = eye_analysis.get('right_ear', 0.0)

                        eyetxt = f"GazeY:{gy:.2f}  Away:{int(la)}  L_EAR:{le:.2f}  R_EAR:{re:.2f}"
                        cv2.putText(annotated_frame, eyetxt, (10, annotated_frame.shape[0] - 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                        # Optionally draw a simple indicator for looking away
                        if la:
                            cv2.putText(annotated_frame, "LOOKING AWAY", (10, annotated_frame.shape[0] - 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Show the combined annotated frame
                    cv2.imshow('AI Interviewer - Camera', annotated_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save summaries for emotion and eye-tracking
                        try:
                            summary = self.emotion_system.analyzer.get_analysis_summary()
                            eye_summary = self.eye_tracker.get_analysis_summary()
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            if summary:
                                with open(f"emotion_summary_{ts}.json", 'w') as f:
                                    json.dump(summary, f, indent=2)
                            if eye_summary:
                                with open(f"eye_summary_{ts}.json", 'w') as f:
                                    json.dump(eye_summary, f, indent=2)
                            print(f"Saved summaries at {ts}")
                        except Exception as e:
                            print(f"Error saving summaries: {e}")

                    # Throttle loop slightly
                    time.sleep(0.01)

            except Exception as e:
                print(f"Error in combined camera loop: {e}")
            finally:
                # Ensure resources are cleaned up
                try:
                    self.emotion_system.stop()
                except Exception:
                    pass
                try:
                    self.eye_tracker.save_log()
                except Exception:
                    pass

        # Start the combined camera thread
        cam_thread = threading.Thread(target=run_combined_camera, daemon=True)
        cam_thread.start()
        return cam_thread
    
    def get_emotion_summary(self):
        """
        Get emotion analysis summary for the current period
        """
        try:
            return self.emotion_system.analyzer.get_analysis_summary()
        except Exception as e:
            print(f"Error getting emotion summary: {e}")
            return None
    
    def conduct_interview(self, num_questions=5, job_role=None):
        """
        Conduct the complete interview process with full voice output
        """
        print("=" * 50)
        print("AI INTERVIEWER - TECHNICAL ROUND")
        print("=" * 50)
        
        # Get available job roles
        available_roles = self.questions_df['Job Role'].unique().tolist()
        roles_text = f"Available job roles are: {', '.join(available_roles)}"
        self.speak_and_print(roles_text)
        
        if not job_role:
            self.speak_and_print("Please select a job role or press Enter for random questions.")
            print("\nSelect a job role or press Enter for random questions:")
            selected_role = input().strip()
            job_role = selected_role if selected_role in available_roles else None
            
            if job_role:
                self.speak_and_print(f"Great! I've selected {job_role} questions for you.")
            else:
                self.speak_and_print("I'll select random questions from all categories for you.")
        
        # Get random questions
        questions = self.get_random_questions(num_questions, job_role)
        
        if not questions:
            self.speak_and_print("No questions available!")
            return
        
        # Initialize voice recognition with the questions
        question_texts = [q['Question'] for q in questions]
        self.voice_recognizer = voice_rec(question_texts)
        
        # Start emotion monitoring
        self.speak_and_print("Starting emotion monitoring system.")
        print("\nStarting emotion monitoring...")
        emotion_thread = self.start_emotion_monitoring()
        time.sleep(2)  # Give camera time to initialize
        
        # Conduct interview
        interview_start_text = f"Starting your technical interview with {len(questions)} questions."
        self.speak_and_print(interview_start_text)
        print(f"\nStarting interview with {len(questions)} questions...")
        
        self.speak_and_print("The interview will begin in 3 seconds. Get ready!")
        print("The interview will begin in 3 seconds...")
        time.sleep(3)
        
        # Introduction
        self.voice_recognizer.speak("Hello! Welcome to your technical interview.")
        self.voice_recognizer.speak("I will ask you several technical questions.")
        self.voice_recognizer.speak("Please answer each question to the best of your ability.")
        self.voice_recognizer.speak("Let's begin!")
        
        # Ask each question
        for i, question_data in enumerate(questions, 1):
            question_header = f"Question {i} of {len(questions)}"
            print(f"\n--- {question_header} ---")
            self.voice_recognizer.speak(question_header)
            
            question = question_data['Question']
            expected_answer = question_data['Answer']
            job_role_q = question_data['Job Role']
            
            print(f"Question: {question}")
            
            # Ask the question using voice - speak the actual question
            self.voice_recognizer.speak("Here is your question:")
            self.voice_recognizer.speak(question)
            self.voice_recognizer.speak("Please provide your answer now.")
            
            user_answer = self.voice_recognizer.listen()
            
            if user_answer:
                response_text = f"Thank you. Your answer was: {user_answer}"
                print(f"Your answer: {user_answer}")
                self.voice_recognizer.speak("Thank you for your response. Let me analyze your answer.")
                
                # Calculate similarity
                similarity_score = self.calculate_similarity(user_answer, expected_answer)
                
                # Get emotion analysis for this question period
                emotion_summary = self.get_emotion_summary()
                
                # Store question data
                question_result = {
                    'question_number': i,
                    'job_role': job_role_q,
                    'question': question,
                    'expected_answer': expected_answer,
                    'user_answer': user_answer,
                    'similarity_score': similarity_score,
                    'emotion_analysis': emotion_summary,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.current_interview['questions_asked'].append(question_result)
                
                # Provide feedback with voice
                score_text = f"Your similarity score is {similarity_score:.2f} out of 1.0"
                print(f"Similarity Score: {similarity_score:.2f}")
                self.voice_recognizer.speak(score_text)
                
                if similarity_score >= 0.7:
                    feedback = "Excellent answer! You demonstrated strong knowledge of this topic."
                elif similarity_score >= 0.5:
                    feedback = "Good answer! You covered the main points well."
                elif similarity_score >= 0.3:
                    feedback = "Fair answer, but there's room for improvement. Consider reviewing this topic further."
                else:
                    feedback = "This topic needs more attention. I recommend studying this area more thoroughly."
                
                print(f"Feedback: {feedback}")
                self.voice_recognizer.speak(feedback)
                
                # Brief pause between questions
                if i < len(questions):
                    self.voice_recognizer.speak("Let's move on to the next question.")
                time.sleep(2)
            else:
                no_answer_text = "I didn't receive an answer for this question. Let's move on."
                print("No answer recorded.")
                self.voice_recognizer.speak(no_answer_text)
        
        # End interview
        self.voice_recognizer.speak("Congratulations! You have completed the technical interview.")
        self.voice_recognizer.speak("Your performance is being analyzed and the results will be saved.")
        
        # Stop emotion monitoring
        self.emotion_system.stop()
        
        # Calculate overall performance
        self.calculate_overall_performance()
        
        # Speak the performance summary
        self.speak_performance_summary()
        
        # Save results
        self.save_interview_results()
        
        completion_text = f"Interview completed! Results have been saved to {self.log_file}"
        print(f"\nInterview completed! Results saved to {self.log_file}")
        self.voice_recognizer.speak("All results have been successfully saved. Thank you for using the AI interviewer!")
    
    def speak_performance_summary(self):
        """
        Speak the performance summary aloud
        """
        if not self.current_interview['overall_performance']:
            return
            
        performance = self.current_interview['overall_performance']
        
        # Speak overall summary
        summary_intro = "Now let me provide you with your interview performance summary."
        self.voice_recognizer.speak(summary_intro)
        
        total_q = f"You answered {performance['total_questions']} questions in total."
        self.voice_recognizer.speak(total_q)
        
        avg_score = f"Your average similarity score is {performance['average_similarity_score']:.2f} out of 1.0"
        self.voice_recognizer.speak(avg_score)
        
        grade_text = f"Your overall performance grade is: {performance['performance_grade']}"
        self.voice_recognizer.speak(grade_text)
        
        confidence_text = f"Your average confidence level was {performance['average_confidence']:.2f}"
        self.voice_recognizer.speak(confidence_text)
        
        nervousness_text = f"Your average nervousness level was {performance['average_nervousness']:.2f}"
        self.voice_recognizer.speak(nervousness_text)
        
        # Speak individual scores
        self.voice_recognizer.speak("Here are your individual question scores:")
        for i, score in enumerate(performance['individual_scores'], 1):
            score_detail = f"Question {i}: {score:.2f}"
            self.voice_recognizer.speak(score_detail)
    
    def calculate_overall_performance(self):
        """
        Calculate overall interview performance metrics
        """
        if not self.current_interview['questions_asked']:
            return
        
        # Calculate average similarity score
        similarity_scores = [q['similarity_score'] for q in self.current_interview['questions_asked']]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Calculate performance grade
        if avg_similarity >= 0.8:
            grade = "Excellent"
        elif avg_similarity >= 0.6:
            grade = "Good"
        elif avg_similarity >= 0.4:
            grade = "Fair"
        else:
            grade = "Needs Improvement"
        
        # Analyze emotions (if available)
        emotion_scores = []
        confidence_scores = []
        nervousness_scores = []
        
        for question in self.current_interview['questions_asked']:
            if question['emotion_analysis']:
                emotion_data = question['emotion_analysis']
                if 'average_emotions' in emotion_data:
                    # Extract confidence and nervousness indicators
                    emotions = emotion_data['average_emotions']
                    confidence = emotions.get('happy', 0) + emotions.get('neutral', 0)
                    nervousness = emotions.get('fear', 0) + emotions.get('sad', 0)
                    
                    confidence_scores.append(confidence)
                    nervousness_scores.append(nervousness)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        avg_nervousness = sum(nervousness_scores) / len(nervousness_scores) if nervousness_scores else 0
        
        self.current_interview['overall_performance'] = {
            'total_questions': len(self.current_interview['questions_asked']),
            'average_similarity_score': avg_similarity,
            'performance_grade': grade,
            'average_confidence': avg_confidence,
            'average_nervousness': avg_nervousness,
            'individual_scores': similarity_scores
        }
    
    def save_interview_results(self):
        """
        Save interview results to JSON file
        """
        try:
            # Load existing interviews if file exists
            existing_interviews = []
            try:
                with open(self.log_file, 'r') as f:
                    existing_interviews = json.load(f)
            except FileNotFoundError:
                pass
            
            # Add current interview
            existing_interviews.append(self.current_interview)
            
            # Save to file
            with open(self.log_file, 'w') as f:
                json.dump(existing_interviews, f, indent=2)
            
            print(f"Interview results saved to {self.log_file}")
            
        except Exception as e:
            print(f"Error saving interview results: {e}")
    
    def display_performance_summary(self):
        """
        Display a summary of the interview performance
        """
        performance = self.current_interview['overall_performance']
        
        print("\n" + "=" * 50)
        print("INTERVIEW PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Total Questions: {performance['total_questions']}")
        print(f"Average Similarity Score: {performance['average_similarity_score']:.2f}")
        print(f"Performance Grade: {performance['performance_grade']}")
        print(f"Average Confidence: {performance['average_confidence']:.2f}")
        print(f"Average Nervousness: {performance['average_nervousness']:.2f}")
        print("\nDetailed Scores:")
        
        for i, score in enumerate(performance['individual_scores'], 1):
            print(f"  Question {i}: {score:.2f}")
        
        print("=" * 50)

def main():
    """
    Main function to run the AI Interviewer
    """
    try:
        # Initialize the AI Interviewer
        interviewer = AIInterviewer()
        
        # Welcome message with voice
        welcome_text = "Welcome to the AI Technical Interviewer!"
        print(welcome_text)
        
        try:
            num_questions = int(input("How many questions would you like? (default: 5): ") or "5")
        except ValueError:
            num_questions = 5
        
        # Conduct the interview
        interviewer.conduct_interview(num_questions=num_questions)
        
        # Display performance summary
        interviewer.display_performance_summary()
        
    except KeyboardInterrupt:
        print("\nInterview interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
