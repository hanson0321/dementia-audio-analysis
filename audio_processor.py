# audio_processor.py

# --- 標準庫導入 ---
import os
import sys
import logging
import time # 用於計時

# --- 第三方庫導入 ---
import pandas as pd
import soundfile as sf # 用於保存音訊
import torch # 可能在處理 waveform 時用到
from tqdm.auto import tqdm # 用於顯示進度條
import numpy as np # 用於 np.nan

# --- 導入我們自己的模塊 ---
try:
    import config
    import metadata_processor # 負責載入和匹配中繼資料
    import audio_utils # 負責音訊處理函數 (載入, 切片, 增強)
except ImportError as e:
    logging.critical(f"錯誤：無法導入必要的模塊 (config, metadata_processor, audio_utils): {e}")
    logging.critical("請確保這些檔案存在於專案根目錄並已正確配置。")
    sys.exit("無法載入內部模塊，程式停止。")

# 設定模塊級別的日誌 (可選，或者依賴 config 中的基本設置)
# logging.basicConfig(level=config.logging.INFO)
# logging.getLogger(__name__).setLevel(config.logging.INFO) # 設定當前模塊的日誌級別

logging.info("audio_processor.py 載入成功。")


def process_audio_files() -> pd.DataFrame | None:
    """
    執行語音檔案的完整處理流程：載入中繼資料，切片，增強，保存片段和中繼資料。

    Returns:
        pd.DataFrame | None: 包含所有處理後語音片段中繼資料的 DataFrame，
                             或 None 如果中繼資料載入或處理過程中發生嚴重錯誤。
    """
    logging.info("\n--- 開始進行語音處理 (切片和資料增強) ---")

    # 1. 載入和匹配中繼資料
    # 調用 metadata_processor 模塊中的函數
    metadata_df = metadata_processor.load_and_match_metadata()

    # 如果中繼資料載入或匹配失敗，則停止
    if metadata_df is None or metadata_df.empty:
        logging.error("錯誤：無法載入或匹配中繼資料。語音處理終止。")
        return None

    logging.info(f"準備處理 {len(metadata_df)} 筆音訊檔案。")

    # 列表用於收集所有生成的片段的中繼資料
    all_chunk_metadata = []

    start_time_processing = time.time() # 開始計時

    # 2. 遍歷每個音訊檔案，進行切片和增強
    # 遍歷 metadata_df 中的每一行，其中包含了原始音訊檔案的完整路徑
    for index, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="處理語音檔案"):
        original_filename = row[config.INTERNAL_FILENAME_COL]
        original_audio_full_path = row['audio_full_path'] # 從 DataFrame 獲取原始音訊檔案的完整路徑
        participant_id = row[config.INTERNAL_ID_COL]
        label = row[config.INTERNAL_LABEL_COL]
        # score = row[config.INTERNAL_SCORE_COL] # 我們不做回歸，這裡不需要 score

        # 檢查原始音訊檔案是否存在 (metadata_processor 已經做過檢查，這裡再次檢查是為了安全)
        if not os.path.exists(original_audio_full_path):
            logging.warning(f"警告：原始音訊檔案不存在: {original_audio_full_path}。跳過處理。")
            continue

        # 載入原始音訊檔案
        waveform = audio_utils.load_audio(original_audio_full_path, target_sr=config.SAMPLE_RATE, target_channels=config.N_CHANNELS)

        if waveform is None:
            logging.warning(f"跳過檔案 {original_filename} 的處理，因為載入失敗。")
            continue # 跳過這個檔案

        #logging.debug(f"成功載入檔案 {original_filename}，長度：{waveform.shape[0]/config.SAMPLE_RATE:.2f} 秒。")

        # 將音訊切片
        chunks = audio_utils.chunk_audio(waveform, config.SAMPLE_RATE, config.CHUNK_DURATION_SEC)
        #logging.debug(f"檔案 {original_filename} 被切分成 {len(chunks)} 個片段。")

        # 處理並保存每個片段
        for i, chunk in enumerate(chunks):
            # 構造切片檔案名：原始檔案名 (不含副檔名) + '_chunk_' + 片段索引 + .wav
            base_filename = os.path.splitext(original_filename)[0] # 例如 0001_林曾秀貞
            chunk_filename_prefix = f"{base_filename}_chunk_{i:03d}" # 使用 3 位數字格式化索引，例如 0001_林曾秀貞_chunk_000

            # --- 保存原始片段 ---
            chunk_filename_original = f"{chunk_filename_prefix}.wav"
            chunk_path_original = os.path.join(config.CHUNKS_DIR, chunk_filename_original)

            try:
                # 確保 chunk 是 shape [samples] 且 dtype 是 float32
                if chunk.ndim == 1 and chunk.dtype == config.AUDIO_DTYPE:
                    # 使用 soundfile 保存為 WAV 檔案
                    sf.write(chunk_path_original, chunk.numpy(), config.SAMPLE_RATE)
                    # logging.debug(f"保存原始片段：{chunk_path_original}") # 日誌太多，暫時註解掉

                    # 添加原始片段的中繼資料
                    all_chunk_metadata.append({
                        config.INTERNAL_FILENAME_COL: chunk_filename_original,
                        config.INTERNAL_ID_COL: participant_id,
                        config.INTERNAL_LABEL_COL: label,
                        config.INTERNAL_SCORE_COL: np.nan, # 分數欄位保持為 NaN
                        'original_audio_file': original_filename,
                        'chunk_index': i,
                        'is_augmented': False, # 這是一個原始片段
                        'start_time_sec': i * config.CHUNK_DURATION_SEC,
                        'end_time_sec': (i + 1) * config.CHUNK_DURATION_SEC,
                        'chunk_audio_path': chunk_path_original # 記錄切片檔案的完整路徑
                    })
                else:
                     logging.warning(f"片段 {chunk_filename_original} 格式異常 (shape: {chunk.shape}, dtype: {chunk.dtype})，跳過保存原始片段。")

            except Exception as e:
                logging.error(f"錯誤：無法保存原始片段 {chunk_path_original}: {e}")
                # 如果保存原始片段失敗，則不再對這個片段進行增強或記錄
                continue # 跳過保存這個片段的增強版本


            # --- 資料增強 (僅對失智症樣本應用，如果功能可用且增強器已初始化) ---
            # 僅對 'Dementia' 類別的樣本應用增強
            # 確保 label 轉換為小寫進行比較，以防 CSV 中大小寫不一致
            if config.AUDIO_AUGMENTATION_AVAILABLE and audio_utils.augmenter is not None and str(label).lower() == 'dementia':
                 # 調用 audio_utils 中的增強函數
                 augmented_chunk = audio_utils.apply_augmentation(chunk, config.SAMPLE_RATE, audio_utils.augmenter)

                 if augmented_chunk is not None:
                     chunk_filename_augmented = f"{chunk_filename_prefix}_aug.wav" # 添加 _aug 標識
                     chunk_path_augmented = os.path.join(config.AUGMENTED_DIR, chunk_filename_augmented)

                     try:
                         # 確保增強後的 waveform 仍然是 shape [samples] 且 dtype 是 float32
                         if augmented_chunk.ndim == 1 and augmented_chunk.dtype == config.AUDIO_DTYPE:
                              sf.write(chunk_path_augmented, augmented_chunk.numpy(), config.SAMPLE_RATE)
                              # logging.debug(f"保存增強片段：{chunk_path_augmented}") # 日誌太多，暫時註解掉

                              # 添加增強片段的中繼資料
                              all_chunk_metadata.append({
                                  config.INTERNAL_FILENAME_COL: chunk_filename_augmented,
                                  config.INTERNAL_ID_COL: participant_id,
                                  config.INTERNAL_LABEL_COL: label,
                                  config.INTERNAL_SCORE_COL: np.nan,
                                  'original_audio_file': original_filename,
                                  'chunk_index': i,
                                  'is_augmented': True, # 這是一個增強片段
                                  'start_time_sec': i * config.CHUNK_DURATION_SEC,
                                  'end_time_sec': (i + 1) * config.CHUNK_DURATION_SEC,
                                  'chunk_audio_path': chunk_path_augmented # 記錄切片檔案的完整路徑
                              })
                         else:
                              logging.warning(f"對片段 {chunk_filename_original} 應用增強後 waveform 格式異常 (shape: {augmented_chunk.shape}, dtype: {augmented_chunk.dtype})，跳過保存增強片段。")


                     except Exception as e:
                         logging.error(f"錯誤：無法保存增強片段 {chunk_path_augmented}: {e}")
                 else:
                      # apply_augmentation 內部已經打印了錯誤，這裡不再重複
                      pass # logging.warning(f"對片段 {chunk_filename_original} 的增強失敗，跳過保存增強片段。")


    # --- 保存所有片段的中繼資料 --- (在處理完所有檔案後執行一次)

    end_time_processing = time.time() # 結束計時
    processing_duration = end_time_processing - start_time_processing
    logging.info(f"\n語音切片和增強處理完成。總計耗時: {processing_duration:.2f} 秒")


    if all_chunk_metadata:
        chunk_metadata_df = pd.DataFrame(all_chunk_metadata)

        try:
            chunk_metadata_df.to_csv(config.CHUNK_METADATA_FILE, index=False)
            logging.info(f"\n成功生成並保存所有語音片段的中繼資料到：{config.CHUNK_METADATA_FILE}")
            logging.info(f"總共生成了 {len(chunk_metadata_df)} 個語音片段 (包含原始和增強)。")
            logging.info("片段中繼資料前 5 行:")
            # 選擇性打印部分欄位以便閱讀
            print(chunk_metadata_df[[config.INTERNAL_FILENAME_COL, config.INTERNAL_ID_COL, config.INTERNAL_LABEL_COL, 'is_augmented', 'chunk_index', 'chunk_audio_path']].head().to_string())

            logging.info(f"\n語音片段類別分佈 ('{config.INTERNAL_LABEL_COL}' 欄位):\n{chunk_metadata_df[config.INTERNAL_LABEL_COL].value_counts().to_string()}")
            # 檢查是否有增強的片段
            if 'is_augmented' in chunk_metadata_df.columns and chunk_metadata_df['is_augmented'].any():
                 logging.info(f"語音片段增強分佈 ('is_augmented' 欄位):\n{chunk_metadata_df['is_augmented'].value_counts().to_string()}")


            print("\n--- 資料處理階段完成 ---")
            print(f"原始語音切片已保存到 '{os.path.abspath(config.CHUNKS_DIR)}'")
            if config.AUDIO_AUGMENTATION_AVAILABLE and audio_utils.augmenter is not None:
                # 檢查 augmented 資料夾是否有檔案
                if any(os.scandir(config.AUGMENTED_DIR)): # 檢查目錄是否為空
                     print(f"增強語音已保存到 '{os.path.abspath(config.AUGMENTED_DIR)}'")
                else:
                     print("未生成增強語音片段 (可能因為沒有失智症樣本或增強失敗)。")
            else:
                 print("由於資料增強功能不可用或未初始化，未生成增強語音。")

            print(f"所有片段的中繼資料已保存到 '{os.path.abspath(config.CHUNK_METADATA_FILE)}'")

            return chunk_metadata_df # 返回生成的片段中繼資料 DataFrame

        except Exception as e:
            logging.error(f"錯誤：無法保存片段中繼資料到 {config.CHUNK_METADATA_FILE}: {e}")
            logging.error("語音片段處理完成，但中繼資料保存失敗。")
            return None # 保存失敗，返回 None

    else:
        # 雖然 metadata_df 不為空，但 all_chunk_metadata 可能為空 (例如所有檔案都載入失敗)
        logging.warning("\n根據中繼資料，應該處理語音檔案，但最終沒有生成任何語音片段。請檢查：")
        logging.warning(f"1. 原始中繼資料包含 {len(metadata_df)} 筆數據，但沒有片段生成。")
        logging.warning(f"2. 檢查 '{config.RAW_AUDIO_DIR}' 中的音訊檔案是否有效且能被載入。")
        print("\n語音片段處理階段未能完成。")
        return None # 沒有生成片段，返回 None


# 如果直接運行此腳本，則執行處理流程
if __name__ == "__main__":
    # 在這裡可以確保 config 先被載入
    if 'config' not in sys.modules:
         logging.critical("錯誤：無法載入 config.py。請確保 config.py 存在於專案根目錄。")
         sys.exit("無法載入配置檔，程式停止。")

    # 確保必要的輸出資料夾存在 (config 已經做了大部分，這裡再次確保)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.CHUNKS_DIR, exist_ok=True)
    os.makedirs(config.AUGMENTED_DIR, exist_ok=True)


    logging.info("--- 正在測試運行 audio_processor.py ---")

    # 調用主處理函數
    processed_df = process_audio_files()

    if processed_df is not None:
        logging.info("\naudio_processor.py 測試運行成功。語音片段處理完成。")
    else:
        logging.error("\naudio_processor.py 測試運行失敗。語音片段處理未能完成。")

    logging.info("--- audio_processor.py 測試運行結束 ---")