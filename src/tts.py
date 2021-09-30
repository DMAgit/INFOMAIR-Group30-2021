import pyttsx3


class TTS:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.rate = 125  # speaking speed
        self.volume = 1.0  # volume [0.0-1.0]
        self.voice = 1  # 0 - male, 1 - female

        self.engine.setProperty("rate", self.rate)
        self.engine.setProperty("volume", self.volume)
        self.engine.setProperty('voice', self.engine.getProperty("voices")[self.voice].id)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
        self.engine.stop()
