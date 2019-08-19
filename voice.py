from gtts import gTTS
import os
from pygame import mixer
mixer.init()
text=("welcome  bala")
speech=gTTS(text,'en','slow')
speech.save("welcome.mp3")
#os.system("welcome.mp3")

mixer.music.load("welcome.mp3")
mixer.music.play()
