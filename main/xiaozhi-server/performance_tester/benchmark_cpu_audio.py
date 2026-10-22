"""对 CPU 音频热点做离线微基准，不连接云 API、不读取密钥。

@author pony
@date 2026-10-22
@version v1.1.0
@last_modified 2026-10-22
@changelog
    - v1.1.0 (2026-10-22): 可选对比真实 TTS PCM → Opus 编码复杂度 5 与 10
    - v1.0.0 (2026-10-22): 测量单线程 Silero 模型延迟及逐包 gzip 耗时和包体
"""

import argparse
import gzip
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
import wave


SERVER_ROOT = Path(__file__).resolve().parents[1]


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return result


def benchmark_gzip(wav_path, iterations):
    with wave.open(str(wav_path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getnchannels() != 1:
            raise ValueError("gzip 基准需要 16 位单声道 WAV")
        raw = audio.readframes(audio.getnframes())
        sample_rate = audio.getframerate()
    # 按音频自身采样率切 60ms 包；生产上送为 16k PCM，非 16k 样本仅用于比较压缩热点。
    packet_size = int(sample_rate * 0.06) * 2
    frames = [raw[index:index + packet_size]
              for index in range(0, len(raw) - packet_size + 1, packet_size)]
    if not frames:
        raise ValueError("WAV 不足 60ms")
    batch = [frames[index % len(frames)] for index in range(iterations)]
    samples = {1: [], 9: []}
    sizes = {}
    rng = random.Random(42)
    for _ in range(5):
        levels = [1, 9]
        rng.shuffle(levels)
        for level in levels:
            start = time.perf_counter()
            packed = [gzip.compress(frame, compresslevel=level) for frame in batch]
            samples[level].append((time.perf_counter() - start) * 1000 / iterations)
            sizes[level] = statistics.mean(map(len, packed))
    return {
        "wav": wav_path.name,
        "sample_rate": sample_rate,
        "packet_ms": 60,
        "packets_per_round": iterations,
        "rounds": 5,
        "levels": {str(level): {
            "median_ms_per_packet": round(statistics.median(samples[level]), 4),
            "mean_bytes_per_packet": round(sizes[level], 1),
        } for level in samples},
    }


def benchmark_model(iterations, threads):
    import torch

    torch.set_num_threads(threads)
    model_path = SERVER_ROOT / "models/snakers4_silero-vad/src/silero_vad/data/silero_vad.jit"
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    frame = torch.zeros(512)
    samples = []
    with torch.no_grad():
        for _ in range(30):
            model(frame, 16000)
        for _ in range(iterations):
            start = time.perf_counter()
            model(frame, 16000)
            samples.append((time.perf_counter() - start) * 1000)
    return {
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "frames": iterations,
        "audio_ms_per_frame": 32,
        "input": "silence",
        "mean_ms_per_frame": round(statistics.mean(samples), 4),
        "p95_ms_per_frame": round(sorted(samples)[int((iterations - 1) * 0.95)], 4),
        "scope": "仅 JIT 模型；不含 Opus、状态切换、线程调度、网络和排队",
    }


def benchmark_opus(wav_path, iterations):
    import numpy as np

    sys.path.insert(0, str(SERVER_ROOT))
    from core.utils.opus_encoder_utils import OpusEncoderUtils

    with wave.open(str(wav_path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getnchannels() != 1:
            raise ValueError("Opus 基准需要 16 位单声道 WAV")
        source_rate = audio.getframerate()
        pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    if len(pcm) < source_rate * 0.06:
        raise ValueError("WAV 不足 60ms")
    # 模拟当前设备常用的 24k TTS；离线重采样不计入编码耗时。
    sample_rate, frame_ms = 24000, 60
    if source_rate != sample_rate:
        positions = np.arange(len(pcm) * sample_rate // source_rate) * source_rate / sample_rate
        pcm = np.interp(positions, np.arange(len(pcm)), pcm).astype("<i2")
    frame_samples = sample_rate * frame_ms // 1000
    frames = [pcm[index:index + frame_samples].tobytes()
              for index in range(0, len(pcm) - frame_samples + 1, frame_samples)]
    samples = {5: [], 10: []}
    cpu_samples = {5: [], 10: []}
    sizes = {}
    rng = random.Random(42)
    for _ in range(5):
        complexities = [5, 10]
        rng.shuffle(complexities)
        for complexity in complexities:
            encoder = OpusEncoderUtils(sample_rate, 1, frame_ms)
            try:
                encoder.set_complexity(complexity)
                for index in range(30):
                    encoder.encode_pcm_to_opus_stream(frames[index % len(frames)], False, lambda _: None)
                packed = []
                cpu_start = time.process_time()
                start = time.perf_counter()
                for index in range(iterations):
                    encoder.encode_pcm_to_opus_stream(frames[index % len(frames)], False, packed.append)
                samples[complexity].append((time.perf_counter() - start) * 1000 / iterations)
                cpu_samples[complexity].append((time.process_time() - cpu_start) * 1000 / iterations)
                if len(packed) != iterations:
                    raise RuntimeError("Opus 基准输出帧数与输入不符")
                sizes[complexity] = statistics.mean(map(len, packed))
            finally:
                encoder.close()
    return {
        "wav": wav_path.name,
        "sample_rate": sample_rate,
        "packet_ms": frame_ms,
        "bitrate": 24000,
        "packets_per_round": iterations,
        "rounds": 5,
        "complexities": {str(level): {
            "median_ms_per_packet": round(statistics.median(samples[level]), 4),
            "median_cpu_ms_per_packet": round(statistics.median(cpu_samples[level]), 4),
            "mean_bytes_per_packet": round(sizes[level], 1),
        } for level in samples},
        "scope": "真实 PCM 缓冲与 Opus 编码；不含网络、重采样、播放、线程调度及音质评估",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=positive_int, default=2000)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--wav", type=Path, default=SERVER_ROOT / "config/assets/wakeup_words_short.wav")
    parser.add_argument("--with-opus", action="store_true", help="额外测量 TTS Opus 编码，需要 opuslib_next 和 libopus")
    args = parser.parse_args()
    result = {
        "environment": {"python": platform.python_version(), "machine": platform.machine(),
                        "logical_cpus": os.cpu_count()},
        "silero_model_only": benchmark_model(args.iterations, args.threads),
        "gzip": benchmark_gzip(args.wav, args.iterations),
        "limits": "微基准不能代表设备并发上限、华为云性能或端到端首音频延迟",
    }
    if args.with_opus:
        result["tts_opus_encode"] = benchmark_opus(args.wav, args.iterations)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
