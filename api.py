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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 确保能找到子模块里的代码
sys.path.append('third_party/Matcha-TTS')

# 初始化FastAPI
app = FastAPI(title="CosyVoice API")

# 加载模型
cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=False,
    load_trt=False,
    fp16=False,
    use_flow_cache=False
)

class SpeechRequest(BaseModel):
    text_instruct: str
    instruction: str
    voice_style: str = "default"

@app.post("/tts")
async def text_to_speech(request: SpeechRequest):
    try:
        # 根据语音风格选择提示音
        if request.voice_style == "zhoujielun":
            prompt_speech = load_wav('./asset/zhoujielun.wav', 16000)
        elif request.voice_style == "nezha":
            prompt_speech = load_wav('./asset/nezha.wav', 16000)
        elif request.voice_style == "linzhiling":
            prompt_speech = load_wav('./asset/linzhiling.wav', 16000)
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
    uvicorn.run(app, host="0.0.0.0", port=50001) 