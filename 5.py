import pyaudio
import time

from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://10.189.3.18:50000/v1", api_key="123")

speech_file_path = Path(__file__).parent / "speech.wav"

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16, channels=1, rate=24000, output=True, frames_per_buffer=8192
)


def tts_streaming(text: str, instructions: str = ""):
    first_chunk_sent = False
    start = time.time()

    with client.audio.speech.with_streaming_response.create(
        model="CosyVoice3",
        voice="Theresa_zh",
        input=text,
        instructions=instructions,
        response_format="wav",
        speed=1.0,
        extra_body={"stream": True},  # TTS流式输出
    ) as response:

        with open(speech_file_path, "wb") as f:
            for chunk in response.iter_bytes(4096):
                if not chunk:
                    continue

                f.write(chunk)  # 写文件
                stream.write(chunk)  # 播放

                if not first_chunk_sent:
                    print(f"首 chunk 延迟: {time.time() - start:.4f} seconds")
                    first_chunk_sent = True

    time.sleep(1)
    stream.stop_stream()
    stream.close()
    p.terminate()


def main():
    text = "我还记得这间会议室，这是专门为特雷西娅空着的位置么？"
    tts_streaming(text=text)


if __name__ == "__main__":
    main()
