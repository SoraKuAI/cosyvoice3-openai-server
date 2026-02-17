import os
os.environ.update({
    "DEEPSPEED_LOG_LEVEL": "error",
    "TOKENIZERS_PARALLELISM": "false",
    "TQDM_DISABLE": "1",
    "PYTHONWARNINGS": "ignore"
})
import io
import sys
import argparse
import random
import uvicorn
import numpy as np
import torch
import soundfile as sf
import lameenc
import time

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from logger import setup_logger

# 第三方路径设置
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{ROOT_DIR}/third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed

# 避免matmul精度问题
torch.set_float32_matmul_precision("high")

# 设置日志
logger = setup_logger(name="CosyVoice3 TTS Server", level="INFO")

# 全局配置
class Config:
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"
    fp16 = False
    enable_trt = True
    enable_vllm = False

config = Config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading CosyVoice model from {config.model_dir}...")
    # 加载 CosyVoice
    app.state.cosyvoice = AutoModel(
        model_dir=config.model_dir,
        fp16=config.fp16,
        load_trt=config.enable_trt,
        load_vllm=config.enable_vllm
    )
    
    # 缓存 Speaker lists
    app.state.available_spks = set(app.state.cosyvoice.list_available_spks())
    
    logger.info("Model Loaded. Ready to serve.")
    logger.info(f"Started server process [{os.getpid()}]")
    logger.info(f"Starting CosyVoice3 OpenAI API server on http://0.0.0.0:{args.port}")
    yield
    
    # 清理资源
    logger.info("Cleaning up...")
    del app.state.cosyvoice
    app.state.available_spks = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

# 跨域解决
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 请求结构
class TTSRequest(BaseModel):
    model: str
    input: str
    voice: str
    instructions: str = ""
    stream: bool = False
    seed: Optional[int] = None
    speed: float = 1.0
    response_format: str = "wav"

# 接口：获取模型列表
@app.get("/v1/models")
async def list_models():
    logger.info("GET /v1/models")
    return {"object": "list", "data": [{"id": "CosyVoice3", "owned_by": "NorthSky"}]}

# 接口：列出所有预训练的说话人
@app.get("/v1/speakers")
async def list_speakers():
    logger.info("GET /v1/speakers")
    return {
        "object": "list", 
        "data": [{"id": s} for s in app.state.available_spks]
    }

# 接口：生成随机数
@app.get("/v1/seed")
async def generate_seed():
    logger.info("Generate seed...")
    return {"seed": random.randint(1, 4294967295)}

# 接口：TTS生成
@app.post("/v1/audio/speech")
async def inference(req: TTSRequest):
    start_time = time.time()

    logger.info(f"POST /v1/audio/speech | model={req.model} voice={req.voice} stream={req.stream}")
    
    # 1. 校验
    if not req.input:
        raise HTTPException(400, "文本不能为空")
    if req.voice not in app.state.available_spks:
        raise HTTPException(400, f"未知说话人: {req.voice}")

    # 构造 Prompt
    instr = req.instructions or ""
    prompt = f"You are a helpful assistant.{instr}<|endofprompt|>{req.input}"

    if req.seed is not None:
        set_all_random_seed(req.seed)
    
    sr = app.state.cosyvoice.sample_rate

    # 流式输出
    if req.stream:
        def generate_stream():
            total_duration = 0
            try:
                for chunk in app.state.cosyvoice.inference_sft(prompt, req.voice, stream=True, speed=req.speed):
                    if (audio := chunk.get("tts_speech")) is not None:
                        # 计算当前分块时长并累加
                        chunk_dur = audio.shape[1] / sr
                        total_duration += chunk_dur
                        
                        yield audio.cpu().numpy().tobytes()
                
                # 流式结束后的统计
                end_time = time.time()
                elapsed = end_time - start_time
                if total_duration > 0:
                    rtf = elapsed / total_duration
                    logger.info(
                        f"Stream finished | total_dur={total_duration:.2f}s | "
                        f"elapsed={elapsed:.2f}s | rtf={rtf:.3f}"
                    )
                else:
                    logger.warning("Stream finished with zero duration")
                    
            except Exception as e:
                logger.error(f"Stream Error: {e}")

        return StreamingResponse(generate_stream(), media_type=f"audio/L32; rate={sr}; channels=1")
    
    # 非流式处理
    else:
        try:
            outputs = list(app.state.cosyvoice.inference_sft(prompt, req.voice, stream=False, speed=req.speed))
            audio_chunks = [c["tts_speech"] for c in outputs if c.get("tts_speech") is not None]
            
            if not audio_chunks:
                raise HTTPException(500, "生成失败")
            
            combined_tensor = torch.cat(audio_chunks, dim=1).squeeze().cpu()
            final_audio_np = combined_tensor.numpy()
            
            # 统计数据计算
            speech_len = len(final_audio_np) / sr
            rtf = (time.time() - start_time) / speech_len
            
            byte_io = io.BytesIO()
            
            if req.response_format == "mp3":
                audio_int16 = (final_audio_np * 32767).clip(-32768, 32767).astype(np.int16)
                enc = lameenc.Encoder()
                enc.set_bit_rate(192)
                enc.set_in_sample_rate(sr)
                enc.set_channels(1)
                byte_io.write(enc.encode(audio_int16.tobytes()) + enc.flush())
                media_type = "audio/mpeg"
            else:
                sf.write(byte_io, final_audio_np, sr, format='WAV', subtype='FLOAT')
                media_type = "audio/wav"
            
            byte_io.seek(0)
            logger.info(f"Done | dur={speech_len:.2f}s | latency={(time.time()-start_time)*1000:.1f}ms | rtf={rtf:.3f}")
            return StreamingResponse(byte_io, media_type=media_type)
        
        except Exception as e:
            logger.exception("Inference failed")
            raise HTTPException(500, f"生成出错: {str(e)}")


if __name__ == "__main__":
    import warnings
    import logging

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.getLogger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="CosyVoice3 OpenAI-Compatible API Server")
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--model_dir", type=str, default="pretrained_models/Fun-CosyVoice3-0.5B")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trt", action="store_true", dest="enable_trt")
    parser.add_argument("--vllm", action="store_true", dest="enable_vllm")
    parser.set_defaults(enable_trt=False, enable_vllm=False)
    args = parser.parse_args()
    
    config.model_dir, config.fp16 = args.model_dir, args.fp16
    config.enable_trt, config.enable_vllm = args.enable_trt, args.enable_vllm

    logger.info(f"Starting CosyVoice3 API | Port: {args.port} | TRT: {config.enable_trt} | FP16: {config.fp16}")

    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=args.port, 
            log_config=None, 
            access_log=False
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception:
        logger.exception("Fatal error during startup")
    finally:
        logger.info("Shutdown complete")
