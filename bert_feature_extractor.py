# bert_feature_extractor.py

# --- 標準庫導入 ---
import os
import sys
import logging
import time

# --- 第三方庫導入 ---
import pandas as pd
import torch
import numpy as np
from tqdm.auto import tqdm

# Hugging Face 生態系統導入
try:
    from transformers import AutoTokenizer, AutoModel
    logging.info("Hugging Face Transformers (AutoTokenizer, AutoModel) 模組導入成功。")
except ImportError as e:
    logging.critical(f"--- 嚴重錯誤：無法導入 BERT 模型相關的 Hugging Face 模組 ---")
    logging.critical(f"詳細錯誤：{e}")
    logging.critical("\n請確認您已在conda環境中並成功安裝套件：")
    logging.critical("pip install transformers torch")
    sys.exit("無法載入 BERT 相關模組，語言特徵提取停止。")

# --- 導入我們自己的模塊 ---
try:
    import config
    # audio_utils 可能不需要直接在此腳本中使用，但 config 會導入它
    logging.info("內部模塊 (config) 導入成功。")
except ImportError as e:
    logging.critical(f"--- 嚴重錯誤：無法導入必要的內部模塊 (config) ---")
    logging.critical(f"詳細錯誤：{e}")
    sys.exit("無法載入內部模塊，程式停止。")

# 重新配置日誌，以避免在多次導入 config 時重複添加 handler
# (如果 config.py 已經配置了 logging.basicConfig)
# for handler in logging.root.handlers[:]:
#    logging.root.removeHandler(handler)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.getLogger("transformers").setLevel(logging.ERROR) # 減少 transformers 的日誌輸出
# logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logging.info("bert_feature_extractor.py 載入成功。")

# --- 主特徵提取函數 ---

def extract_bert_features_batched(batch_size: int):
    """
    執行批次處理的 BERT 語言特徵提取。
    會讀取 features_metadata.csv 中的轉錄，並更新該檔案以包含語言特徵路徑。

    Args:
        batch_size (int): 每次處理的轉錄文本數量。

    Returns:
        pd.DataFrame | None: 更新後的 features_metadata DataFrame，
                             或 None 如果處理過程中發生嚴重錯誤。
    """
    logging.info(f"\n--- 開始批次處理 BERT 語言特徵提取 (批次大小: {batch_size}) ---")

    # 1. 檢查並載入 features_metadata.csv
    if not os.path.exists(config.FEATURES_METADATA_FILE):
        logging.error(f"錯誤：未找到特徵中繼資料檔案: {config.FEATURES_METADATA_FILE}。無法進行語言特徵提取。")
        logging.error("請先成功運行 whisper_feature_extractor.py。")
        return None

    try:
        features_df = pd.read_csv(config.FEATURES_METADATA_FILE)
        logging.info(f"\n成功載入特徵中繼資料檔案: {config.FEATURES_METADATA_FILE}")
        logging.info(f"共 {len(features_df)} 筆記錄 (包含聲學特徵和轉錄)。")
        if features_df.empty:
             logging.warning("警告：載入的特徵中繼資料檔案為空。沒有轉錄可處理。")
             return None
        if 'transcription' not in features_df.columns:
            logging.error(f"錯誤：特徵中繼資料檔案 '{config.FEATURES_METADATA_FILE}' 缺少 'transcription' 欄位。")
            return None
    except Exception as e:
        logging.error(f"錯誤：無法載入特徵中繼資料檔案 {config.FEATURES_METADATA_FILE}: {e}")
        return None

    # 2. 準備斷點續傳：檢查 'linguistic_feature_path' 欄位
    linguistic_feature_col = 'linguistic_feature_path'
    if linguistic_feature_col not in features_df.columns:
        features_df[linguistic_feature_col] = pd.NA # 使用 pd.NA 以更好地區分空值
        logging.info(f"已在 DataFrame 中添加新的 '{linguistic_feature_col}' 欄位。")
    else:
        # 確保現有欄位中的空字串或純空格也被視為 NaN/NA，以便重新處理 (如果需要)
        features_df[linguistic_feature_col] = features_df[linguistic_feature_col].replace(r'^\s*$', pd.NA, regex=True)


    # 找出未處理的記錄 (linguistic_feature_path 為空或 NaN/pd.NA)
    unprocessed_df = features_df[features_df[linguistic_feature_col].isnull()].copy() # 使用 .copy() 避免 SettingWithCopyWarning

    logging.info(f"還有 {len(unprocessed_df)} 個轉錄文本待提取語言特徵。")

    if unprocessed_df.empty:
        logging.info("\n所有轉錄文本的語言特徵似乎都已提取完畢。")
        return features_df # 返回已完整的 DataFrame

    # 3. 獲取當前批次要處理的記錄
    current_batch_to_process_df = unprocessed_df.head(batch_size)
    logging.info(f"將處理當前批次的 {len(current_batch_to_process_df)} 個轉錄文本。")

    if current_batch_to_process_df.empty:
        logging.info("\n當前批次沒有轉錄文本可處理。")
        return features_df

    # 4. 載入 BERT 模型和 Tokenizer (只有需要處理時才載入)
    bert_tokenizer = None
    bert_model = None
    try:
        logging.info(f"\n正在載入 BERT 模型 '{config.BERT_MODEL_NAME}' 和 Tokenizer...")
        bert_tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        logging.info("BERT Tokenizer 載入成功。")
        bert_model = AutoModel.from_pretrained(config.BERT_MODEL_NAME).to(config.DEVICE)
        bert_model.eval() # 設置為評估模式
        logging.info(f"BERT 模型 '{config.BERT_MODEL_NAME}' 載入成功到 {config.DEVICE}。")
        logging.info(f"BERT 模型參數數量: {bert_model.num_parameters()/1e6:.2f}M")
    except Exception as e:
        logging.error(f"錯誤：BERT 模型或 Tokenizer 載入過程中發生異常: {e}")
        logging.error(f"請檢查模型名稱 '{config.BERT_MODEL_NAME}', 網絡連接, Hugging Face Token (若需)。")
        # 如果模型載入失敗，我們不應該修改 features_df，直接返回它
        return features_df

    # 5. 處理當前批次的轉錄文本
    processed_linguistic_paths_batch = {} # 用於存儲 {original_dataframe_index: path_or_NA}
    logging.info("\n--- 處理當前批次的轉錄文本 ---")
    start_time_processing_batch = time.time()

    # 提取轉錄文本列表和它們在原始 DataFrame 中的索引
    transcriptions_batch = []
    original_indices_batch = []

    for original_idx, row_series in current_batch_to_process_df.iterrows():
        transcription_text = row_series['transcription']
        # 處理 NaN 或 Whisper 失敗標記
        if pd.isna(transcription_text) or not isinstance(transcription_text, str) or \
           transcription_text.strip() == "" or transcription_text.startswith("["):
            logging.warning(
                f"警告：索引 {original_idx} (檔案名: {row_series.get(config.INTERNAL_FILENAME_COL, 'N/A')}) "
                f"的轉錄文本為空、NaN 或標記為失敗 ('{transcription_text}')。將生成零向量特徵。"
            )
            # 對於這些情況，我們仍然會生成一個 .npy 檔案（包含零向量）
            # 並將文本視為空字串傳給 tokenizer (tokenizer 通常能處理空字串)
            transcriptions_batch.append("") # Tokenizer 可以處理空字串
        else:
            transcriptions_batch.append(transcription_text)
        original_indices_batch.append(original_idx)


    # --- Tokenize 和提取特徵 ---
    if transcriptions_batch: # 確保列表不為空
        try:
            inputs = bert_tokenizer(transcriptions_batch,
                                    padding=True,
                                    truncation=True,
                                    max_length=bert_tokenizer.model_max_length if hasattr(bert_tokenizer, 'model_max_length') else 512,
                                    return_tensors="pt"
                                   ).to(config.DEVICE)

            with torch.no_grad():
                outputs = bert_model(**inputs)
                # 使用 pooler_output (對應 [CLS] token 經過線性層和 Tanh 激活)
                # BERT pooler_output 是為句子級別任務設計的
                linguistic_features_batch = outputs.pooler_output # Shape: (batch_size, hidden_size)
                # 或者，如果想用 last_hidden_state 的平均池化:
                # last_hidden_states = outputs.last_hidden_state
                # attention_mask = inputs['attention_mask']
                # masked_hidden_states = last_hidden_states * attention_mask.unsqueeze(-1)
                # sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
                # sum_attention_mask = torch.sum(attention_mask, dim=1, keepdim=True)
                # linguistic_features_batch = sum_hidden_states / sum_attention_mask

            linguistic_features_batch_np = linguistic_features_batch.cpu().numpy()

            # 保存每個特徵並記錄路徑
            for i, original_idx in enumerate(original_indices_batch):
                # 獲取對應的原始 DataFrame 行來構造檔案名
                # current_row = features_df.loc[original_idx] # 這樣更安全
                current_row = current_batch_to_process_df.loc[original_idx]

                chunk_filename = current_row[config.INTERNAL_FILENAME_COL]
                base_chunk_filename = os.path.splitext(chunk_filename)[0]
                ling_feature_filename = f"{base_chunk_filename}_ling.npy"
                ling_feature_path = os.path.join(config.LINGUISTIC_FEATURES_DIR, ling_feature_filename)

                # 獲取當前實際用於 tokenization 的文本
                current_text_for_tokenization = transcriptions_batch[i]
                feature_to_save_np = linguistic_features_batch_np[i]

                # 如果原始轉錄是空的或失敗的，我們之前已經將其設為空字串給 tokenizer
                # tokenizer 對空字串的處理通常會產生一個基於 [CLS] 和 [SEP] 的有效嵌入
                # 如果我們想明確地為這些情況保存全零向量，可以這樣做：
                original_transcription_text = current_batch_to_process_df.loc[original_idx, 'transcription']
                if pd.isna(original_transcription_text) or not isinstance(original_transcription_text, str) or \
                   original_transcription_text.strip() == "" or original_transcription_text.startswith("["):
                    feature_to_save_np = np.zeros_like(feature_to_save_np) # 使用提取到的特徵的形狀創建零向量


                try:
                    np.save(ling_feature_path, feature_to_save_np)
                    processed_linguistic_paths_batch[original_idx] = ling_feature_path
                except Exception as e_save:
                    logging.error(f"錯誤：無法保存語言特徵檔案 {ling_feature_path} for index {original_idx}: {e_save}")
                    processed_linguistic_paths_batch[original_idx] = pd.NA

        except Exception as e_batch_process:
            logging.error(f"錯誤：在批次處理語言特徵時發生異常: {e_batch_process}")
            for original_idx in original_indices_batch:
                processed_linguistic_paths_batch[original_idx] = pd.NA
    else:
        logging.info("當前批次的轉錄文本列表為空，無需處理。")


    end_time_processing_batch = time.time()
    batch_processing_duration = end_time_processing_batch - start_time_processing_batch
    if transcriptions_batch: # 只有實際處理了才打印耗時
        logging.info(f"\n當前批次 ({len(transcriptions_batch)}個轉錄) 的語言特徵提取完成。總計耗時: {batch_processing_duration:.2f} 秒")

    # 6. 更新 features_df 中的 linguistic_feature_path
    if processed_linguistic_paths_batch:
        for original_idx, path_or_na in processed_linguistic_paths_batch.items():
            features_df.loc[original_idx, linguistic_feature_col] = path_or_na
        logging.info(f"已更新 DataFrame 中 {len(processed_linguistic_paths_batch)} 筆記錄的 '{linguistic_feature_col}'。")

        try:
            features_df.to_csv(config.FEATURES_METADATA_FILE, index=False)
            logging.info(f"\n成功更新並保存特徵中繼資料檔案到：{config.FEATURES_METADATA_FILE}")
            logging.info("更新後的特徵中繼資料前 5 行 (部分欄位):")
            print(features_df[[config.INTERNAL_FILENAME_COL, 'transcription', 'acoustic_feature_path', linguistic_feature_col]].head().to_string())

            # 統計已完成和剩餘
            remaining_count = features_df[linguistic_feature_col].isnull().sum()
            total_count = len(features_df)
            processed_count = total_count - remaining_count

            logging.info(f"\n總計記錄: {total_count}")
            logging.info(f"已提取語言特徵: {processed_count}")
            logging.info(f"剩餘待提取: {remaining_count}")

            if remaining_count > 0:
                 print(f"請再次運行此腳本以處理剩餘的 {remaining_count} 個轉錄文本。")
            else:
                 print("所有轉錄文本的語言特徵都已提取完畢。")
                 logging.info("\n接下來可以準備模型訓練了。")

            return features_df

        except Exception as e_save_csv:
            logging.error(f"錯誤：無法保存更新後的特徵中繼資料到 {config.FEATURES_METADATA_FILE}: {e_save_csv}")
            logging.error("當前批次處理完成，但中繼資料保存失敗。下次運行時將會重新處理此批次的記錄。")
            return None # 保存失敗
    else:
        # 如果 processed_linguistic_paths_batch 為空 (例如 transcriptions_batch 為空)
        # 或者所有處理都失敗了，沒有新的路徑可以更新
        if not unprocessed_df.empty and not current_batch_to_process_df.empty and not transcriptions_batch:
             # 這種情況是批次中有東西但都被過濾為空文本了
             logging.info("當前批次所有轉錄文本均為空或無效，沒有生成新的語言特徵。")
        else:
             logging.warning("\n當前批次沒有生成任何語言特徵數據 (可能是所有轉錄處理失敗或批次為空)。")
        return features_df # 返回原始或部分更新的 DataFrame


# 如果直接運行此腳本
if __name__ == "__main__":
    # 確保必要的輸出資料夾存在
    try:
        os.makedirs(config.LINGUISTIC_FEATURES_DIR, exist_ok=True)
    except AttributeError:
        logging.error("錯誤：config 模塊中 LINGUISTIC_FEATURES_DIR 未定義。")
        sys.exit("配置錯誤，程式停止。")
    except Exception as e:
        logging.error(f"錯誤：創建輸出目錄 '{config.LINGUISTIC_FEATURES_DIR}' 失敗: {e}")
        sys.exit("目錄創建失敗，程式停止。")

    logging.info(f"--- 正在運行 {os.path.basename(__file__)} ---")

    # === 在這裡指定批次大小 ===
    # BERT 模型通常比 Whisper 小，但文本 tokenization 和處理也需要資源
    # 根據您的 VRAM/RAM 調整
    # current_bert_batch_size = 16 # 或 32, 64
    current_bert_batch_size = getattr(config, 'BERT_BATCH_SIZE', 16) # 從 config 或使用預設值
    logging.info(f"將使用批次大小: {current_bert_batch_size}")
    # =========================

    updated_features_df_result = extract_bert_features_batched(batch_size=current_bert_batch_size)

    if updated_features_df_result is not None:
        logging.info(f"\n{os.path.basename(__file__)} 批次運行結束。")
        remaining_to_process_count = updated_features_df_result[updated_features_df_result['linguistic_feature_path'].isnull()].shape[0]
        if remaining_to_process_count == 0:
            logging.info("所有語言特徵均已成功提取並記錄！")
        elif not updated_features_df_result.empty :
            logging.info(f"還有 {remaining_to_process_count} 個語言特徵待處理，請再次運行腳本。")
        else: # DataFrame is None or empty
             logging.info("批次運行結束，但沒有生成或更新語言特徵數據。")
    else:
        logging.error(f"\n{os.path.basename(__file__)} 批次運行遇到嚴重錯誤導致中斷或保存失敗。請檢查日誌。")

    logging.info(f"--- {os.path.basename(__file__)} 運行結束 ---")