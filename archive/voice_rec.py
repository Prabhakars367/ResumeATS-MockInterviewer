import speech_recognition as sr
import pyttsx3
import time


class voice_rec:
    def __init__(self, questions=None):
        """
        Initialize voice recognition system
        """
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.engine = pyttsx3.init()
        self.questions = questions or []
        
        # Configure speech recognition settings
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 1
        self.recognizer.phrase_threshold = 0.3
        
        # Configure text-to-speech settings
        self.engine.setProperty('rate', 150)  # Speaking rate
        self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        
        # Test microphone on initialization
        self._test_microphone()
    
    def _test_microphone(self):
        """
        Test if microphone is working properly
        """
        try:
            with self.microphone as source:
                print("Testing microphone... Please wait.")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Microphone test completed successfully.")
        except Exception as e:
            print(f"Warning: Microphone test failed: {e}")
    
    def speak(self, text):
        """
        Convert text to speech
        """
        try:
            print(f"AI: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
            time.sleep(0.5)  # Brief pause after speaking
        except Exception as e:
            print(f"TTS Error: {e}")
            print(f"AI: {text}")  # Fallback to text display
    
    def listen(self, prompt=None, timeout=10, phrase_time_limit=15, max_retries=3):
        """
        Listen for user input with proper timeout and error handling
        
        Args:
            prompt: Question or prompt to speak before listening
            timeout: Maximum time to wait for speech to start (seconds)
            phrase_time_limit: Maximum time for the user to speak (seconds)
            max_retries: Maximum number of retry attempts
        """
        if prompt:
            self.speak(prompt)
        
        retries = 0
        while retries < max_retries:
            try:
                with self.microphone as source:
                    if retries == 0:
                        print("Adjusting for ambient noise... Please wait.")
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    
                    print(f"Listening... (Attempt {retries + 1}/{max_retries})")
                    print("Speak now, or press Ctrl+C to type your answer instead.")
                    
                    # Listen with timeout
                    audio = self.recognizer.listen(
                        source, 
                        timeout=timeout, 
                        phrase_time_limit=phrase_time_limit
                    )
                
                # Try to recognize the speech
                print("Processing your speech...")
                response = self.recognizer.recognize_google(audio, language='en-US')
                
                if response.strip():
                    print(f"You said: {response}")
                    return response.strip()
                else:
                    print("Empty response detected.")
                    retries += 1
                    
            except sr.WaitTimeoutError:
                print(f"No speech detected within {timeout} seconds.")
                retries += 1
                if retries < max_retries:
                    self.speak("I didn't hear anything. Please try again.")
                
            except sr.UnknownValueError:
                print("Could not understand the audio clearly.")
                retries += 1
                if retries < max_retries:
                    self.speak("Sorry, I couldn't understand that. Please speak more clearly.")
                
            except sr.RequestError as e:
                print(f"Speech recognition service error: {e}")
                retries += 1
                if retries < max_retries:
                    self.speak("Having trouble with speech recognition. Please try again.")
                
            except KeyboardInterrupt:
                print("\nSpeech input interrupted. You can type your answer instead.")
                return input("Type your answer: ").strip()
                
            except Exception as e:
                print(f"Unexpected error during speech recognition: {e}")
                retries += 1
        
        # If all retries failed, offer text input as fallback
        print("Speech recognition failed after multiple attempts.")
        self.speak("I'm having trouble hearing you. Please type your answer instead.")
        return input("Type your answer: ").strip()
    
    def listen_with_fallback(self, prompt=None):
        """
        Listen with immediate fallback option
        """
        if prompt:
            self.speak(prompt)
        
        print("\nChoose input method:")
        print("1. Press Enter to speak your answer")
        print("2. Type 't' to type your answer")
        
        choice = input("Your choice (Enter for speech, 't' for text): ").strip().lower()
        
        if choice == 't':
            return input("Type your answer: ").strip()
        else:
            return self.listen(timeout=15, phrase_time_limit=30, max_retries=2)
    
    def start_interview(self):
        """
        Legacy method for backward compatibility
        """
        self.speak("Hello, I am your interviewer for this session.")
        self.speak("Please introduce yourself.")
        
        intro = self.listen_with_fallback("Tell me about yourself.")
        answers = {"introduction": intro}
        
        for i, question in enumerate(self.questions, 1):
            self.speak(f"Question {i}: {question}")
            answer = self.listen_with_fallback()
            answers[f"question_{i}"] = answer
            
        return answers
    
    def test_system(self):
        """
        Test the voice recognition system
        """
        print("Testing voice recognition system...")
        self.speak("Voice recognition test started.")
        
        test_response = self.listen(
            "Please say 'Hello, this is a test' to verify the system is working.",
            timeout=10,
            max_retries=2
        )
        
        if test_response and "test" in test_response.lower():
            self.speak("Voice recognition test passed successfully!")
            print("✓ Voice recognition system is working properly.")
            return True
        else:
            print("✗ Voice recognition test failed or was unclear.")
            return False


# Test function to verify the system works
def test_voice_rec():
    """
    Test function to verify voice recognition works properly
    """
    test_questions = [
        "What is your favorite programming language?",
        "Tell me about a project you've worked on."
    ]
    
    voice_system = voice_rec(test_questions)
    
    # Run system test
    if voice_system.test_system():
        print("System ready for interview!")
    else:
        print("Please check your microphone setup.")


if __name__ == "__main__":
    test_voice_rec()
