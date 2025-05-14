# audio_utils.py

# --- 標準庫導入 ---
import os
import logging
import math # 需要用到 math.ceil

# --- 第三方庫導入 ---
import torch
import torchaudio
import numpy as np
import soundfile as sf
# audiomentations 會被條件性導入

# --- 從 config.py 導入設定 ---
try:
    import config
except ImportError:
    # 在 utils 模塊中，如果 config 導入失敗通常是嚴重的
    logging.critical("錯誤：無法導入 config.py。請確保 config.py 存在於專案根目錄並已正確配置。")
    # 在獨立模塊中直接 sys.exit 不太好，更好的方式是在調用處處理導入錯誤
    # 但為了確保 config 可用性，這裡可以打印嚴重錯誤並讓上層調用者決定如何處理
    pass # 上層調用者會檢查 config 是否成功導入


# 初始化 Audiomentations 增強器 (如果庫可用且已初始化)
# 這個 augmenter 對象在模塊載入時初始化一次
augmenter = None
if config.AUDIO_AUGMENTATION_AVAILABLE:
    try:
        import audiomentations as audioaug
        # 您可以根據您的需求調整增強策略和參數
        # 示例策略：加雜訊、變速、變調、時間位移
        # 在 config.py 中我們已經設置了 AUDIO_AUGMENTATION_AVAILABLE 標誌
        augmenter = audioaug.Compose(
            transforms=[
                audioaug.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5), # 50% 機率加高斯雜訊
                audioaug.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5, leave_length_unchanged=False), # 50% 機率變速 (0.8x 到 1.2x)
                audioaug.PitchShift(min_semitones=-2, max_semitones=2, p=0.5, sample_rate=config.SAMPLE_RATE), # 50% 機率變調 (-2 到 +2 半音)
                audioaug.Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5), # 50% 機率時間位移
                # 您可以添加更多增強或調整參數
            ],
            p=1.0 # 應用 Compose 中至少一個變換的總機率設為 100%
        )
        logging.info("Audiomentations 增強器初始化成功。")
    except ImportError:
        # 這個 ImportError 應該已經在 config.py 處理了，這裡再次捕獲是為了安全
        logging.warning("警告：無法導入 audiomentations 庫，增強器初始化失敗。")
        augmenter = None
    except Exception as e:
         logging.warning(f"警告：初始化 Audiomentations 增強器失敗: {e}")
         augmenter = None

logging.info("audio_utils.py 載入成功。")


def load_audio(file_path: str, target_sr: int = config.SAMPLE_RATE, target_channels: int = config.N_CHANNELS) -> torch.Tensor | None:
    """
    載入音訊檔案，轉換取樣率、聲道和 dtype。

    Args:
        file_path (str): 音訊檔案的完整路徑。
        target_sr (int): 目標取樣率 (Hz)。
        target_channels (int): 目標聲道數 (1 為 mono)。

    Returns:
        torch.Tensor | None: 處理後的音訊 waveform (shape: [samples]) 或 None 如果載入失敗。
    """
    try:
        # torchaudio.load 返回 waveform (tensor) 和原始取樣率
        waveform, sample_rate = torchaudio.load(file_path)

        # 轉換取樣率 (如果需要)
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)

        # 轉換聲道 (如果需要)
        if waveform.shape[0] > target_channels:
            # 如果是立體聲轉單聲道，簡單取平均
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif waveform.shape[0] < target_channels:
            # 如果是單聲道轉立體聲 (通常不需要)
            logging.warning(f"警告: 檔案 {os.path.basename(file_path)} 聲道數 {waveform.shape[0]} 小於目標聲道數 {target_channels}。跳過處理。")
            return None

        # 確保聲道數正確 (應為 [1, samples])
        waveform = waveform[:target_channels, :]

        # 確保 dtype 是 float32
        waveform = waveform.to(dtype=config.AUDIO_DTYPE)

        return waveform.squeeze(0) # 移除聲道維度，變為 [samples]

    except Exception as e:
        logging.error(f"錯誤：無法載入或處理音訊檔案 {os.path.basename(file_path)}: {e}")
        return None


def chunk_audio(waveform: torch.Tensor, sample_rate: int, chunk_duration_sec: int) -> list[torch.Tensor]:
    """
    將音訊 waveform 切割成固定長度的片段，並進行零填充。

    Args:
        waveform (torch.Tensor): 音訊 waveform (shape: [samples])。
        sample_rate (int): 音訊的取樣率 (Hz)。
        chunk_duration_sec (int): 每個片段的長度 (秒)。

    Returns:
        list[torch.Tensor]: 包含各個音訊片段 waveform (shape: [CHUNK_SAMPLE_SIZE]) 的列表。
    """
    chunk_sample_size = int(chunk_duration_sec * sample_rate)
    total_samples = waveform.shape[0]
    num_chunks = math.ceil(total_samples / chunk_sample_size) # 使用 ceil 確保包含最後一個不完整片段

    chunks = []
    for i in range(num_chunks):
        start_sample = i * chunk_sample_size
        end_sample = min((i + 1) * chunk_sample_size, total_samples)
        chunk = waveform[start_sample:end_sample]

        # 如果是最後一個片段且不足長度，進行零填充
        if chunk.shape[0] < chunk_sample_size:
             padding_needed = chunk_sample_size - chunk.shape[0]
             # 在張量末尾填充零
             chunk = torch.nn.functional.pad(chunk, (0, padding_needed)) # pad (left, right)

        chunks.append(chunk)

    return chunks


def apply_augmentation(waveform: torch.Tensor, sample_rate: int, augmenter) -> torch.Tensor | None:
    """
    應用資料增強到音訊 waveform。

    Args:
        waveform (torch.Tensor): 音訊 waveform (shape: [samples])。
        sample_rate (int): 音訊的取樣率 (Hz)。
        augmenter: Audiomentations Compose 對象，或 None。

    Returns:
        torch.Tensor | None: 增強後的 waveform (shape: [samples]) 或 None 如果增強失敗或增強器不可用。
    """
    if augmenter is None:
        # logging.warning("資料增強功能不可用，跳過增強。") # 這個警告在調用處打印
        return None

    try:
        # Audiomentations 需要 NumPy 陣列， dtype 應為 float32
        waveform_np = waveform.numpy()
        # Audiomentations 期望 input shape 是 (samples,) 或 (samples, channels)
        # 我們傳入的是 (samples,)
        augmented_waveform_np = augmenter(samples=waveform_np, sample_rate=sample_rate)

        # 確保返回的 tensor 也是 shape [samples] 且 dtype 是 float32
        return torch.from_numpy(augmented_waveform_np).to(dtype=config.AUDIO_DTYPE).squeeze()

    except Exception as e:
        # logging.error(f"錯誤：應用資料增強失敗: {e}") # 這個錯誤在調用處打印
        return None


# 可以添加一個簡單的測試區塊
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- 正在測試運行 audio_utils.py ---")

    # 創建一個模擬 waveform (50 秒長)
    sr = config.SAMPLE_RATE
    chunk_dur = config.CHUNK_DURATION_SEC
    simulated_waveform = torch.randn(int(50 * sr), dtype=config.AUDIO_DTYPE)

    logging.info(f"模擬原始 waveform 長度: {simulated_waveform.shape[0]} 樣本 ({simulated_waveform.shape[0]/sr:.2f} 秒)")

    # 測試切片
    chunks = chunk_audio(simulated_waveform, sr, chunk_dur)
    logging.info(f"切分成 {len(chunks)} 個片段，每個片段期望 {int(chunk_dur * sr)} 樣本。")
    for i, chunk in enumerate(chunks):
        logging.info(f"  片段 {i}: shape {chunk.shape}")
        if chunk.shape[0] != int(chunk_dur * sr):
            logging.error(f"  警告：片段 {i} 長度異常！")

    # 測試增強 (如果可用)
    if augmenter is not None:
        logging.info("測試應用增強...")
        test_chunk_to_augment = chunks[0].clone() # 複製一個片段進行測試
        augmented_chunk = apply_augmentation(test_chunk_to_augment, sr, augmenter)
        if augmented_chunk is not None:
            logging.info(f"  增強後的片段 shape: {augmented_chunk.shape}, dtype: {augmented_chunk.dtype}")
        else:
            logging.warning("  增強測試失敗或增強器不可用。")
    else:
        logging.warning("Audiomentations 增強器未初始化，跳過增強測試。")


    logging.info("--- audio_utils.py 測試運行結束 ---")