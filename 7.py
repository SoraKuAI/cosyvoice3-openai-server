import requests
import pyaudio
import wave

url = "http://192.168.1.200:50000/v1/audio/speech"

payload = {
    "model": "CosyVoice3",
    # "voice": "blny",
    # "voice": "Theresa_jp",
    # "input": "欢迎来到我们新的一期ATM10冒险之旅",
    # "input": "アーミヤのオモイがあったからこそ、ワタシはこのスガタでアラワレタの。",
    # "input": "ムカシムカシ、といってもせいぜいニジュウネンぐらいマエのことなのだけれど、ボクはあるガクセイリョウにスンデイタ。ボクはジュウハチで、ダイガクにハイッタばかりだった。トウキョウのことなんてナニヒトツシラナカッタし、ヒトリぐらしをするのもハジメテだったので、オヤがシンパイしてそのリョウをみつけてきてくれた。そこならショクジもついているし、いろんなセツビもソロッテいるし、セケンシラズのジュウハチのショウネンでもなんとかイキテイケルだろうということだった。もちろんヒヨウのこともあった。",
    "input": "Whether 60 or 16, there is in every human being’s heart the lure of wonders, the unfailing appetite for what’s next and the joy of the game of living. In the center of your heart and my heart, there is a wireless station; so long as it receives messages of beauty, hope, courage and power from man and from the infinite, so long as you are young.",
    # "input": "比如你最叻",
    "stream": True,
    "instructions": "",
    "seed": "114514"
}

headers = {"Content-Type": "application/json", "Authorization": "Bearer 123"}

pcm_data = bytearray()

# 初始化 PyAudio
p = pyaudio.PyAudio()

try:
    with requests.post(url, json=payload, headers=headers, stream=True) as r:
        r.raise_for_status()
        
        sample_rate = int(r.headers.get("X-Sample-Rate", 24000))
        channels = int(r.headers.get("X-Channels", 1))
        bit_depth = int(r.headers.get("X-Bit-Depth", 16))

        # 打开音频输出流
        output_stream = p.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=sample_rate,
            output=True,
        )

        print("正在同步接收并播放...")

        for chunk in r.iter_content(
            chunk_size=1024
        ):  # 较小的 chunk 可以降低首字响应延迟
            if chunk:
                output_stream.write(chunk)
                pcm_data.extend(chunk)

        print("播放完成。")

finally:
    if "output_stream" in locals():
        output_stream.stop_stream()
        output_stream.close()
    p.terminate()

with wave.open(r"./tts_audios/test_pcm_stream.wav", "wb") as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(bit_depth // 8)
    wf.setframerate(sample_rate)
    wf.writeframes(pcm_data)

print("PCM 流保存完成：output/test_pcm_stream.wav")
