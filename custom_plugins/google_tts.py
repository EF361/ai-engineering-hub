# your_custom_plugins/google_tts.py

from google.cloud import texttospeech
import tempfile
import os
import pathlib
import asyncio
import simpleaudio as sa  # or use pydub, pygame, or whatever audio lib you want

class TTS:
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()

    def synthesize(self, text, lang="en-US", voice_name="en-US-Wavenet-D", speaking_rate=1.0):
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang,
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=speaking_rate
        )

        response = self.client.synthesize_speech(
            input=input_text,
            voice=voice,
            audio_config=audio_config
        )

        return response.audio_content

    async def stream(self, text):
        audio = self.synthesize(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out:
            out.write(audio)
            out.flush()
            wave_obj = sa.WaveObject.from_wave_file(out.name)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            os.unlink(out.name)
