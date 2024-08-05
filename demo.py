import torch
import time
from tango2.tango import Tango
import soundfile as sf

# 强制使用 CPU
torch.backends.mps.enabled = False
device = torch.device("cpu")

def generate_nature_sound(prompt, output_file=None):
    print(f"使用设备: {device}")

    start_time = time.time()
    
    # 初始化 Tango 模型，明确指定使用 CPU
    # tango = Tango("declare-lab/tango2-full", device=device)
    tango = Tango("declare-lab/tango-full-ft-audiocaps", device=device)
    # tango = Tango("declare-lab/declare-lab/tango-music-af-ft-mc", device=device)
    
    model_load_time = time.time() - start_time
    print(f"Tango 模型加载完成，耗时 {model_load_time:.2f} 秒")

    print(f"开始生成音频: '{prompt}'")
    generation_start_time = time.time()

    # 生成音频
    audio = tango.generate(prompt)

    generation_time = time.time() - generation_start_time
    print(f"音频生成完成，耗时 {generation_time:.2f} 秒")

    # 如果指定了输出文件，保存音频
    if output_file:
        sf.write(output_file, audio, samplerate=16000)
        print(f"音频已保存为 {output_file}")

    total_time = time.time() - start_time
    print(f"总处理时间（包括模型加载）: {total_time:.2f} 秒")
    print(f"纯生成时间: {generation_time:.2f} 秒")

    return audio

# 使用示例
prompt = "The sound of the shop door being pushed open and the footsteps of a young man."
output_file = f"{prompt}.wav"
generate_nature_sound(prompt, output_file)