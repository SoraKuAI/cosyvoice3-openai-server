import requests

url = "http://192.168.1.200:50000/v1/audio/speech"

payload = {
    "model": "CosyVoice3",
    "voice": "Theresa_zh",
    "input": "我还记得这间会议室，这是专门为特雷西娅空着的位置么？",
    "instructions": "",
    "response_format": "wav",
    "speed": 1.0,
    "stream": False,
}
headers = {"Content-Type": "application/json", "Authorization": "Bearer 123"}

with requests.post(url, json=payload, headers=headers, stream=True) as r:
    r.raise_for_status()
    with open("test_http_stream.wav", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

print("保存完成：test_http_stream.wav")
