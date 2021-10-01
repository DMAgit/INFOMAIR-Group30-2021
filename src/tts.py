import pyttsx3


class TTS:
    def __init__(self):
        self.setup = False
        try:
            self.engine = pyttsx3.init()  # uses the default TTS of the system if present

            self.rate = 125  # speaking speed
            self.volume = 1.0  # volume [0.0-1.0]
            self.voice = 1  # 0 - male, 1 - female

            self.engine.setProperty("rate", self.rate)
            self.engine.setProperty("volume", self.volume)
            self.engine.setProperty('voice', self.engine.getProperty("voices")[self.voice].id)

            self.setup = True  # successfully created
        except Exception:
            print("Your system doesn't support TTS, check drivers (macOS: nsss, windows: sapi5, others: espeak)")

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
        self.engine.stop()
