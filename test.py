import sys
# 确保能找到子模块里的代码
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2 # 注意是 CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# --- CosyVoice 2.0 使用示例 ---

# 加载模型，这里用了推荐的 CosyVoice2-0.5B
# 参数可以调：load_jit/load_trt 控制是否加载优化后的模型，fp16 半精度加速，use_flow_cache 也是加速选项
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)

# 准备一段提示音 (Prompt Speech)，就是你想克隆的声音样本
# 需要是 16kHz 采样率的 WAV 文件，可以用自己的录音替换 './asset/zero_shot_prompt.wav'
prompt_speech_16k = load_wav('./asset/nezha.wav', 16000)

# 要合成的文本
text_to_speak = '我命由我不由天，是魔是仙，我自己说了算。'
# 另一段参考文本 (影响韵律风格)
prompt_text = '前前后后改了八十稿，现在跟我说，还是最初那版好，我去，喂，老表，下来撸串了，巨无霸烤乳猪摊见，敖炳啊，我算是体会到什么叫人间疾苦了，甲方的要求比我闹海还难应付，天劫那是明刀明枪对着干，唉算了，不说了，先干了这杯再说。'

print(f"开始使用提示音 '{'./asset/nezha.wav'}' 合成文本...")

# 调用 zero_shot 推理
# stream=False 表示非流式，一次性生成整段语音
# 返回的是一个生成器，可能有多段结果（比如根据标点自动切分了）
# for i, output in enumerate(cosyvoice.inference_zero_shot(text_to_speak, prompt_text, prompt_speech_16k, stream=False)):
#     # 获取合成的语音数据 (PyTorch Tensor)
#     tts_speech = output['tts_speech']
#     # 保存成 WAV 文件
#     output_filename = f'zero_shot_output_{i}.wav'
#     torchaudio.save(output_filename, tts_speech, cosyvoice.sample_rate)
#     print(f"成功合成第 {i+1} 段语音，已保存为 {output_filename}")

# print("Zero-shot 合成完成！")

# --- 如果想试试方言或特殊效果 (Instruct 模式) ---
text_instruct = '我命由我不由天，是魔是仙，我自己说了算！若前方无路，我便踏出一条路！ 若天地不容，我便扭转这钱坤！'
instruction = '用四川话说。' # 或者 '用粤语说这句话', '用开心的语气说' 等
for i, output in enumerate(cosyvoice.inference_instruct2(text_instruct, instruction, prompt_speech_16k, stream=False)):
    torchaudio.save(f'instruct_output_{i}.wav', output['tts_speech'], cosyvoice.sample_rate)
    print(f"成功合成 Instruct 模式语音，已保存为 instruct_output_{i}.wav")