import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import io
import torch
import time
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cosyvoice_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保能找到子模块里的代码
sys.path.append('third_party/Matcha-TTS')

# 初始化FastAPI
app = FastAPI(title="CosyVoice API")

# 加载模型
logger.info("正在加载语音合成模型...")
cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=False,
    load_trt=False,
    fp16=False,
    use_flow_cache=False
)
logger.info("模型加载完成！")

class SpeechRequest(BaseModel):
    text_instruct: str
    instruction: str
    voice_style: str = "default"

@app.post("/tts")
async def text_to_speech(request: SpeechRequest):
    start_time = time.time()
    logger.info(f"收到新的语音合成请求: {request.text_instruct[:50]}...")
    try:
        # 根据语音风格选择提示音
        logger.info(f"使用语音风格: {request.voice_style}")
        if request.voice_style == "jielun":
            prompt_speech = load_wav('./asset/jielun.wav', 16000)
        elif request.voice_style == "nezha":
            prompt_speech = load_wav('./asset/nezha.wav', 16000)
        elif request.voice_style == "zhiling":
            prompt_speech = load_wav('./asset/zhiling.wav', 16000)
        
        logger.info("开始语音合成...")
        # 合成语音
        outputs = list(cosyvoice.inference_instruct2(
            request.text_instruct,
            request.instruction,
            prompt_speech,
            stream=False
        ))

        if not outputs:
            raise HTTPException(status_code=500, detail="语音合成失败")

        # 合并所有语音片段
        combined_speech = torch.cat([output['tts_speech'] for output in outputs], dim=1)
        
        # 将语音保存到内存缓冲区
        buffer = io.BytesIO()
        torchaudio.save(buffer, combined_speech, cosyvoice.sample_rate, format="wav")
        buffer.seek(0)
        
        duration = time.time() - start_time
        logger.info(f"语音合成完成！耗时: {duration:.2f}秒")
        
        return Response(
            content=buffer.getvalue(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="synthesized_speech.wav"'
            }
        )

    except Exception as e:
        logger.error(f"语音合成错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("启动 CosyVoice API 服务器...")
    uvicorn.run(app, host="0.0.0.0", port=50001) 