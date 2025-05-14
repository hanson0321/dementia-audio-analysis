# whisper_feature_extractor.py (實現智能批次處理和斷點續傳)

# --- 標準庫導入 ---
import os
import sys
import logging
import time # 用於計時

# --- 第三方庫導入 ---
import pandas as pd
import torch
import torchaudio # 載入音訊時需要
import numpy as np # 用於保存特徵為 .npy
from tqdm.auto import tqdm # 用於顯示進度條

# Hugging Face 生態系統導入 (需要 transformers 和 huggingface_hub)
try:
    # 從 transformers 導入 Whisper 模型和 Processor 相關的類別
    # AutoModelForSpeechSeq2Seq 包含了 Encoder 和 Decoder，可以做 ASR
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    # AutoFeatureExtractor, AutoTokenizer 通常包含在 AutoProcessor 中，但保留以防萬一
    logging.info("Hugging Face Transformers (Whisper) 模組導入嘗試成功。")
except ImportError as e:
    logging.critical(f"--- 嚴重錯誤：無法導入 Whisper 模型相關的 Hugging Face 模組 ---")
    logging.critical(f"詳細錯誤：{e}")
    logging.critical("\n請確認您已在conda環境中並成功安裝套件：")
    logging.critical("conda activate your_dementia_env") # 替換為您的環境名稱
    logging.critical("pip install transformers huggingface_hub openai-whisper torch torchaudio") # 確保這些庫已安裝
    sys.exit("無法載入 Whisper 模型相關模組，特徵提取停止。")

# --- 導入我們自己的模塊 ---
try:
    import config
    import audio_utils # 假設 audio_utils 存在且可能被 config 或其他地方使用
    logging.info("內部模塊 (config, audio_utils) 導入成功。")
except ImportError as e:
    logging.critical(f"--- 嚴重錯誤：無法導入必要的內部模塊 (config 或 audio_utils) ---")
    logging.critical(f"詳細錯誤：{e}")
    logging.critical("請確保 config.py 和 audio_utils.py 存在於與此腳本相同的目錄中。")
    sys.exit("無法載入內部模塊，程式停止。")

logging.info("whisper_feature_extractor.py 載入成功。")

# --- 主特徵提取函數 ---

def extract_whisper_features_batched(batch_size: int):
    """
    執行批次處理的 Whisper 聲學特徵提取和語音轉文字。
    會檢查已生成的 features_metadata.csv 來續傳。

    Args:
        batch_size (int): 每次處理的語音片段數量。

    Returns:
        pd.DataFrame | None: 包含所有處理後片段特徵和轉錄中繼資料的 DataFrame (包括之前和當前批次)，
                             或 None 如果處理過程中發生嚴重錯誤導致無法繼續或保存。
    """
    logging.info(f"\n--- 開始批次處理 Whisper 特徵提取和語音轉文字 (批次大小: {batch_size}) ---")

    # 1. 檢查並載入完整的語音片段中繼資料 (所有要處理的片段列表)
    if not os.path.exists(config.CHUNK_METADATA_FILE):
        logging.error(f"錯誤：未找到語音片段中繼資料檔案: {config.CHUNK_METADATA_FILE}。無法進行特徵提取。")
        return None

    try:
        chunk_metadata_df = pd.read_csv(config.CHUNK_METADATA_FILE)
        logging.info(f"\n成功載入完整的語音片段中繼資料檔案: {config.CHUNK_METADATA_FILE}")
        logging.info(f"總共有 {len(chunk_metadata_df)} 個片段待處理。")
        if chunk_metadata_df.empty:
             logging.warning("警告：載入的語音片段中繼資料檔案為空。沒有片段可處理。")
             return None
    except Exception as e:
        logging.error(f"錯誤：無法載入語音片段中繼資料檔案 {config.CHUNK_METADATA_FILE}: {e}")
        return None

    # 2. 載入現有的特徵中繼資料 (已處理的片段列表)
    existing_features_df = pd.DataFrame()
    if os.path.exists(config.FEATURES_METADATA_FILE):
        try:
            existing_features_df = pd.read_csv(config.FEATURES_METADATA_FILE)
            logging.info(f"成功載入現有的特徵中繼資料檔案: {config.FEATURES_METADATA_FILE}")
            logging.info(f"已處理並記錄了 {len(existing_features_df)} 個片段。")
            if config.INTERNAL_FILENAME_COL not in existing_features_df.columns:
                 logging.warning(f"警告：現有特徵中繼資料檔案 '{config.FEATURES_METADATA_FILE}' 缺少內部檔案名欄位 '{config.INTERNAL_FILENAME_COL}'。將重新處理所有片段。")
                 existing_features_df = pd.DataFrame()
        except Exception as e:
            logging.error(f"錯誤：無法載入或解析現有的特徵中繼資料檔案 {config.FEATURES_METADATA_FILE}: {e}")
            logging.warning("將視為沒有已處理的片段，重新處理所有。")
            existing_features_df = pd.DataFrame()

    # 3. 找出未處理的片段
    if not existing_features_df.empty and config.INTERNAL_FILENAME_COL in existing_features_df.columns:
        processed_filenames = existing_features_df[config.INTERNAL_FILENAME_COL].tolist()
    else:
        processed_filenames = []
    logging.info(f"已處理的片段數量 (基於現有中繼資料): {len(processed_filenames)}")

    unprocessed_metadata_df = chunk_metadata_df[
        ~chunk_metadata_df[config.INTERNAL_FILENAME_COL].isin(processed_filenames)
    ].reset_index(drop=True)

    logging.info(f"還有 {len(unprocessed_metadata_df)} 個片段待處理。")

    if unprocessed_metadata_df.empty:
        logging.info("\n所有語音片段似乎都已處理完畢。特徵提取階段完成。")
        return existing_features_df

    # 4. 獲取當前批次要處理的片段
    current_batch_df = unprocessed_metadata_df.head(batch_size).copy()
    logging.info(f"將處理當前批次的 {len(current_batch_df)} 個片段。")

    if current_batch_df.empty: # 理論上如果 unprocessed_metadata_df 不空，這裡也不會空，但雙重檢查
        logging.info("\n當前批次沒有片段可處理 (可能是因為所有片段已處理或 batch_size 問題)。特徵提取階段完成。")
        return existing_features_df

    # 5. 載入 Whisper 模型和 Processor (只有需要處理片段時才載入模型)
    whisper_processor = None
    whisper_model = None
    try:
        logging.info(f"\n正在載入 Whisper 模型 '{config.WHISPER_MODEL_NAME}' 和 Processor...")
        whisper_processor = AutoProcessor.from_pretrained(config.WHISPER_MODEL_NAME)
        logging.info("Whisper Processor 載入成功。")
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(config.WHISPER_MODEL_NAME).to(config.DEVICE)
        logging.info(f"Whisper 模型 '{config.WHISPER_MODEL_NAME}' (AutoModelForSpeechSeq2Seq) 載入成功到 {config.DEVICE}。")
        logging.info(f"Whisper 模型參數數量: {whisper_model.num_parameters()/1e6:.2f}M")
    except Exception as e:
        logging.error(f"錯誤：Whisper 模型或 Processor 載入過程中發生異常: {e}")
        logging.error(f"請檢查模型名稱 '{config.WHISPER_MODEL_NAME}', 網絡連接, Hugging Face Token (若需)。")
        return existing_features_df # 返回已有的數據，本次處理失敗

    # 6. 處理當前批次的語音片段
    all_features_list_current_batch = []
    logging.info("\n--- 處理當前批次的語音片段 ---")
    start_time_processing_batch = time.time()

    for index, row in tqdm(current_batch_df.iterrows(), total=len(current_batch_df), desc=f"處理批次 ({len(current_batch_df)}個)"):
        chunk_filename = row[config.INTERNAL_FILENAME_COL]
        chunk_audio_path = row['chunk_audio_path']
        participant_id = row[config.INTERNAL_ID_COL]
        label = row[config.INTERNAL_LABEL_COL]
        is_augmented = row.get('is_augmented', False) # 使用 .get 提供默認值
        original_audio_file = row.get('original_audio_file', 'N/A')
        chunk_index = row.get('chunk_index', -1)

        # 初始化此片段的結果變數
        acoustic_feature_path_for_this_item = None
        transcription_for_this_item = "[處理失敗]"
        input_features = None # 在每個片段開始時重置

        # --- 載入切片音訊 ---
        try:
            waveform, sr = torchaudio.load(chunk_audio_path)
            if sr != config.SAMPLE_RATE:
                 resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=config.SAMPLE_RATE)
                 waveform = resampler(waveform)
            if waveform.shape[0] > config.N_CHANNELS: # 假設 config.N_CHANNELS = 1
                 waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = waveform.squeeze(0).to(dtype=config.AUDIO_DTYPE)
        except Exception as e:
            logging.error(f"錯誤：無法載入片段音訊檔案 {chunk_audio_path}: {e}")
            transcription_for_this_item = "[載入失敗]"
            all_features_list_current_batch.append({
                config.INTERNAL_FILENAME_COL: chunk_filename, config.INTERNAL_ID_COL: participant_id,
                config.INTERNAL_LABEL_COL: label, config.INTERNAL_SCORE_COL: np.nan,
                'is_augmented': is_augmented, 'original_audio_file': original_audio_file,
                'chunk_index': chunk_index, 'acoustic_feature_path': acoustic_feature_path_for_this_item,
                'transcription': transcription_for_this_item, 'chunk_audio_path': chunk_audio_path
            })
            continue # 跳過此片段的後續處理

        # --- 準備 Whisper 輸入特徵 ---
        try:
             inputs = whisper_processor(waveform.cpu().numpy(), sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
             input_features = inputs.input_features.to(config.DEVICE)
        except Exception as e:
             logging.error(f"錯誤：為片段 {chunk_filename} 準備 Whisper 輸入特徵失敗: {e}")
             input_features = None # 確保標記為 None
             transcription_for_this_item = "[輸入特徵失敗]"
             all_features_list_current_batch.append({
                config.INTERNAL_FILENAME_COL: chunk_filename, config.INTERNAL_ID_COL: participant_id,
                config.INTERNAL_LABEL_COL: label, config.INTERNAL_SCORE_COL: np.nan,
                'is_augmented': is_augmented, 'original_audio_file': original_audio_file,
                'chunk_index': chunk_index, 'acoustic_feature_path': acoustic_feature_path_for_this_item,
                'transcription': transcription_for_this_item, 'chunk_audio_path': chunk_audio_path
            })
             continue # 跳過此片段的後續處理

        # --- 提取聲學特徵 (Whisper Encoder Hidden States) 和執行語音轉文字 (ASR) ---
        if input_features is not None: # 只有當 input_features 成功準備時才執行
            try:
                # --- 提取聲學特徵 ---
                try:
                    encoder = whisper_model.get_encoder()
                    with torch.no_grad():
                        encoder_outputs = encoder(input_features)
                    acoustic_features = encoder_outputs.last_hidden_state.squeeze(0) # [seq_len, hidden_size]

                    base_chunk_filename = os.path.splitext(chunk_filename)[0]
                    feature_filename = f"{base_chunk_filename}.npy"
                    potential_feature_path = os.path.join(config.ACOUSTIC_FEATURES_DIR, feature_filename)
                    np.save(potential_feature_path, acoustic_features.cpu().numpy())
                    acoustic_feature_path_for_this_item = potential_feature_path
                except Exception as acoustic_e:
                    logging.error(f"錯誤：從片段 {chunk_filename} 提取聲學特徵失敗: {acoustic_e}")
                    acoustic_feature_path_for_this_item = None # 確保失敗時為 None

                # --- 執行 ASR ---
                transcription_for_this_item = "[轉錄失敗]" # 預設為失敗
                try:
                    generate_kwargs = {"max_new_tokens": 256} # 可以根據需要調整
                    # decoder_start_token_id 通常由模型配置自動處理，但在某些情況下可能需要明確指定
                    # if hasattr(whisper_model.config, 'decoder_start_token_id') and whisper_model.config.decoder_start_token_id is not None:
                    #     generate_kwargs["decoder_start_token_id"] = whisper_model.config.decoder_start_token_id
                    # elif hasattr(whisper_processor.tokenizer, 'bos_token_id') and whisper_processor.tokenizer.bos_token_id is not None:
                    #      generate_kwargs["decoder_start_token_id"] = whisper_processor.tokenizer.bos_token_id
                    # else:
                    #      logging.warning(f"無法確定 decoder_start_token_id for {chunk_filename}。ASR 可能仍能工作。")

                    predicted_ids = whisper_model.generate(inputs=input_features, **generate_kwargs)
                    transcription_for_this_item = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                except Exception as asr_e:
                    logging.error(f"錯誤：從片段 {chunk_filename} 進行 ASR 失敗: {asr_e}")
                    transcription_for_this_item = "[轉錄失敗]" # 確保失敗時標記

            except Exception as e: # 捕獲此 try 塊內的未預期錯誤
                logging.error(f"錯誤：處理片段 {chunk_filename} (特徵提取/ASR階段) 時發生未預期的異常: {e}")
                acoustic_feature_path_for_this_item = None
                transcription_for_this_item = "[處理異常]"
        else: # input_features is None (已在前一個 try-except 中處理並記錄)
            # transcription_for_this_item 已經被設為 "[輸入特徵失敗]"
            pass


        # --- 收集此片段的最終結果 (無論成功或失敗) ---
        all_features_list_current_batch.append({
            config.INTERNAL_FILENAME_COL: chunk_filename,
            config.INTERNAL_ID_COL: participant_id,
            config.INTERNAL_LABEL_COL: label,
            config.INTERNAL_SCORE_COL: np.nan, # 分數欄位，此處未計算，置為 NaN
            'is_augmented': is_augmented,
            'original_audio_file': original_audio_file,
            'chunk_index': chunk_index,
            'acoustic_feature_path': acoustic_feature_path_for_this_item,
            'transcription': transcription_for_this_item,
            'chunk_audio_path': chunk_audio_path
        })
    # <<< --- FOR 迴圈結束 --- >>>

    end_time_processing_batch = time.time()
    batch_processing_duration = end_time_processing_batch - start_time_processing_batch
    logging.info(f"\n當前批次 ({len(current_batch_df)}個片段) 的聲學特徵提取和語音轉文字處理完成。總計耗時: {batch_processing_duration:.2f} 秒")

    # --- 保存當前批次的特徵和轉錄中繼資料 ---
    if all_features_list_current_batch:
        current_batch_features_df = pd.DataFrame(all_features_list_current_batch)

        # 確保 'score' 欄位存在，以防萬一 (雖然上面已添加)
        if config.INTERNAL_SCORE_COL not in current_batch_features_df.columns:
             current_batch_features_df[config.INTERNAL_SCORE_COL] = np.nan

        combined_features_df = pd.concat([existing_features_df, current_batch_features_df], ignore_index=True)

        try:
            combined_features_df.to_csv(config.FEATURES_METADATA_FILE, index=False)
            logging.info(f"\n成功保存當前批次的特徵和轉錄中繼資料，並附加到：{config.FEATURES_METADATA_FILE}")
            logging.info(f"目前總共記錄了 {len(combined_features_df)} 個片段。")
            logging.info("合併後的特徵中繼資料前 5 行 (部分欄位):")
            print(combined_features_df[[config.INTERNAL_FILENAME_COL, 'transcription', 'acoustic_feature_path']].head().to_string())

            logging.info("\n當前批次轉錄文本前 5 例 (若成功):")
            successful_transcriptions_current_batch = current_batch_features_df[
                ~current_batch_features_df['transcription'].str.startswith("[") # 更通用的失敗檢測
            ].copy()
            if not successful_transcriptions_current_batch.empty:
                sample_transcriptions = successful_transcriptions_current_batch.head(min(5, len(successful_transcriptions_current_batch)))
                for j, r in sample_transcriptions.iterrows():
                     logging.info(f"  {j+1}. ({r[config.INTERNAL_FILENAME_COL]}): {r['transcription']}")
            else:
                logging.warning("當前批次所有片段的轉錄都失敗或未完成，沒有成功轉錄的文本示例。")

            print("\n--- 當前批次處理完成 ---")
            total_processed_count = len(combined_features_df)
            total_to_process_count = len(chunk_metadata_df)
            remaining_count = total_to_process_count - total_processed_count
            logging.info(f"總計待處理片段 (來自 chunk_metadata): {total_to_process_count}")
            logging.info(f"已處理片段總數 (寫入 features_metadata): {total_processed_count}")
            logging.info(f"剩餘待處理片段: {remaining_count}")

            if remaining_count > 0:
                 print(f"請再次運行此腳本以處理剩餘的 {remaining_count} 個片段。")
            else:
                 print("所有語音片段都已處理完畢。特徵提取階段完成。")
                 logging.info("\n接下來可以進行語言特徵提取 (例如使用 BERT)。")

            return combined_features_df

        except Exception as e:
            logging.error(f"錯誤：無法保存合併後的特徵和轉錄中繼資料到 {config.FEATURES_METADATA_FILE}: {e}")
            logging.error("當前批次處理完成，但中繼資料保存失敗。下次運行時將會重新處理此批次的片段。")
            return None # 保存失敗，返回 None 表示本次運行未完全成功記錄
    else:
        logging.warning("\n當前批次沒有生成任何特徵或轉錄數據 (可能是所有片段處理失敗)。請檢查日誌。")
        # 返回已有的數據，因為沒有新的成功數據被添加
        return existing_features_df


# 如果直接運行此腳本，則執行批次特徵提取流程
if __name__ == "__main__":
    # 確保必要的輸出資料夾存在
    try:
        os.makedirs(config.ACOUSTIC_FEATURES_DIR, exist_ok=True)
        os.makedirs(config.LINGUISTIC_FEATURES_DIR, exist_ok=True) # 儘管此腳本不用，但通常一起創建
    except AttributeError:
        logging.error("錯誤：config 模塊中 ACOUSTIC_FEATURES_DIR 或 LINGUISTIC_FEATURES_DIR 未定義。")
        sys.exit("配置錯誤，程式停止。")
    except Exception as e:
        logging.error(f"錯誤：創建輸出目錄失敗: {e}")
        sys.exit("目錄創建失敗，程式停止。")


    logging.info(f"--- 正在運行 {os.path.basename(__file__)} ---")

    # === 在這裡指定批次大小 ===
    # 建議從一個較小的值開始測試，例如 2 或 5，然後根據您的硬體資源逐漸增加。
    # 如果遇到 CUDA out of memory，請減小批次大小。
    # current_batch_size = 2 # 用於快速測試
    current_batch_size = getattr(config, 'WHISPER_BATCH_SIZE', 100) # 從 config 或使用預設值
    logging.info(f"將使用批次大小: {current_batch_size}")
    # =========================

    features_df_result = extract_whisper_features_batched(batch_size=current_batch_size)

    if features_df_result is not None:
        logging.info(f"\n{os.path.basename(__file__)} 批次運行結束。")
        if not features_df_result.empty:
             logging.info(f"目前總共處理並記錄了 {len(features_df_result)} 個片段的特徵。")
        elif os.path.exists(config.FEATURES_METADATA_FILE):
             logging.info("批次運行結束，但沒有新的片段被處理 (可能所有片段已處理完畢)。")
        else:
             logging.info("批次運行結束，但沒有生成任何特徵數據 (可能是首次運行且遇到問題)。")

        # 檢查是否所有片段都已處理
        if os.path.exists(config.CHUNK_METADATA_FILE) and os.path.exists(config.FEATURES_METADATA_FILE):
            try:
                chunk_meta = pd.read_csv(config.CHUNK_METADATA_FILE)
                feat_meta = pd.read_csv(config.FEATURES_METADATA_FILE)
                if len(chunk_meta) == len(feat_meta):
                    logging.info("所有語音片段均已成功處理完畢！")
                elif len(feat_meta) < len(chunk_meta) :
                    logging.info(f"還有 {len(chunk_meta) - len(feat_meta)} 個片段待處理，請再次運行腳本。")
                else: # len(feat_meta) > len(chunk_meta) 不太可能發生除非 chunk_metadata 被修改
                    logging.warning("已處理的片段數量大於總片段數量，請檢查中繼資料檔案。")
            except Exception as e:
                logging.error(f"檢查處理進度時出錯: {e}")
    else:
        logging.error(f"\n{os.path.basename(__file__)} 批次運行遇到嚴重錯誤導致中斷或保存失敗。請檢查日誌中的錯誤訊息。")

    logging.info(f"--- {os.path.basename(__file__)} 運行結束 ---")