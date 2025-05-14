# metadata_processor.py

# --- 標準庫導入 ---
import os
import sys
import logging
import glob # 用於尋找檔案
import re # 用於正則表達式匹配檔案名 (可選)

# --- 第三方庫導入 ---
import pandas as pd
from tqdm.auto import tqdm # 用於顯示進度條
import numpy as np # 用於 np.nan

# --- 從 config.py 導入設定 ---
try:
    import config
except ImportError:
    logging.critical("錯誤：無法導入 config.py。請確保 config.py 存在於專案根目錄並已正確配置。")
    sys.exit("無法載入配置檔，程式停止。")

# 設定模塊級別的日誌 (可選，或者依賴 config 中的基本設置)
# logging.basicConfig(level=config.logging.INFO)
# logging.getLogger(__name__).setLevel(config.logging.INFO) # 設定當前模塊的日誌級別

logging.info("metadata_processor.py 載入成功。")

def load_and_match_metadata() -> pd.DataFrame | None:
    """
    載入原始中繼資料檔案，尋找匹配的音訊檔案，並生成標準化的 DataFrame。

    Returns:
        pd.DataFrame | None: 包含標準化中繼資料的 DataFrame (filename, participant_id, label, score, audio_full_path)
                             或 None 如果載入或匹配失敗。
    """
    logging.info("\n--- 載入中繼資料並匹配音訊檔案 ---")

    metadata_input_df = None
    metadata_df = None # 初始化最終用於處理的 DataFrame

    # 檢查 metadata.csv 檔案是否存在
    if not os.path.exists(config.METADATA_FILE):
        logging.error(f"錯誤：未找到中繼資料檔案: {config.METADATA_FILE}")
        logging.error(f"請將您的 metadata.csv 檔案放置於正確的路徑: {os.path.abspath(config.METADATA_FILE)}")
        return None # 載入失敗，返回 None

    try:
        # 讀取您的 CSV 檔案，指定實際的欄位名稱
        metadata_input_df = pd.read_csv(config.METADATA_FILE)
        logging.info(f"成功載入原始中繼資料檔案: {config.METADATA_FILE}")
        logging.info(f"共 {len(metadata_input_df)} 筆原始資料。")
        logging.info(f"原始中繼資料欄位: {metadata_input_df.columns.tolist()}")

        # 檢查原始 CSV 是否包含必要的輸入欄位
        required_input_cols = [config.INPUT_ID_COL, config.INPUT_LABEL_COL]
        if not all(col in metadata_input_df.columns for col in required_input_cols):
            logging.error(f"錯誤：原始中繼資料檔案 '{config.METADATA_FILE}' 缺少必要的輸入欄位。需要包含: {required_input_cols}")
            return None # 缺少欄位，返回 None

    except Exception as e:
        logging.error(f"錯誤：無法載入或解析原始中資料檔案 {config.METADATA_FILE}: {e}")
        return None # 載入或解析失敗，返回 None


    # 如果原始中繼資料成功載入，則進一步處理和匹配音訊檔案
    logging.info("正在匹配音訊檔案和中繼資料...")

    # 讀取 RAW_AUDIO_DIR 中的所有 .wav 檔案
    all_audio_files_paths = glob.glob(os.path.join(config.RAW_AUDIO_DIR, "*.wav"))
    audio_filenames_in_dir = [os.path.basename(f) for f in all_audio_files_paths]
    logging.info(f"在 '{config.RAW_AUDIO_DIR}' 中找到 {len(audio_filenames_in_dir)} 個 .wav 檔案。")

    # 創建一個列表來存放最終匹配後的資料
    final_metadata_list = []
    matched_audio_paths = {} # 字典用於快速查找音訊檔案的完整路徑 {filename: full_path}
    for p in all_audio_files_paths:
        matched_audio_paths[os.path.basename(p)] = p

    # 遍歷原始中繼資料 DataFrame 的每一行
    # 這裡使用 iterrows()，對於大型 DataFrame 可能效率不高，但對於 1000 筆數據應該足夠
    # 如果數據量非常大，可以考慮優化，例如先創建 filename 查找表
    for index, row in tqdm(metadata_input_df.iterrows(), total=len(metadata_input_df), desc="匹配檔案"):
        participant_id_raw = row[config.INPUT_ID_COL]
        label_raw = row[config.INPUT_LABEL_COL]

        # 將受試者編號轉換為字串，並嘗試進行零填充以匹配檔案名格式
        participant_id_str = str(participant_id_raw)

        matching_files = []
        try:
            # 嘗試零填充匹配 (適用於數字 ID)，假設 ID 是 4 位數字零填充
            participant_id_padded = participant_id_str.zfill(4)
            # 尋找以零填充 ID 開頭的檔案 (例如 '0001_...') 或精確匹配零填充 ID + .wav (例如 '0001.wav')
            matching_files = [f for f in audio_filenames_in_dir if f.startswith(participant_id_padded + '_') or f == participant_id_padded + '.wav']

            # 如果上面沒有找到，並且原始 ID 字串本身可能就是檔案名的一部分 (例如非數字 ID)
            if not matching_files and participant_id_str != participant_id_padded:
                 # 嘗試直接匹配原始 ID 字串開頭
                 matching_files = [f for f in audio_filenames_in_dir if f.startswith(participant_id_str + '_') or f == participant_id_str + '.wav']


        except Exception:
            # 如果在處理 ID 時出現任何異常，使用原始字串嘗試匹配
            matching_files = [f for f in audio_filenames_in_dir if f.startswith(participant_id_str + '_') or f == participant_id_str + '.wav']


        if len(matching_files) == 1:
            filename = matching_files[0]
            # 檢查檔案名是否以 '.' 結尾 (可能是沒有副檔名的情況，雖然不常見但防範一下)
            if filename.endswith('.'):
                 logging.warning(f"警告：檔案名 '{filename}' 格式異常 (以 '.' 結尾)。跳過此檔案。")
                 continue # 跳過這個檔案

            # 檢查是否在 matched_audio_paths 中找到完整路徑 (理論上 glob 找到的都在這裡)
            if filename not in matched_audio_paths:
                 logging.error(f"嚴重錯誤：匹配到的檔案名 '{filename}' 未在 glob 列表中找到完整路徑。跳過此檔案。")
                 continue


            full_audio_path = matched_audio_paths[filename]

            final_metadata_list.append({
                config.INTERNAL_FILENAME_COL: filename, # 例如 0001_林曾秀貞.wav
                config.INTERNAL_ID_COL: participant_id_str, # 例如 '1'
                config.INTERNAL_LABEL_COL: label_raw, # 例如 'Normal' 或 'Dementia'
                config.INTERNAL_SCORE_COL: np.nan, # 分數欄位保持為 NaN，因為我們不做回歸
                'audio_full_path': full_audio_path # 添加音訊檔案的完整路徑，方便後續載入
            })
        elif len(matching_files) > 1:
            logging.warning(f"警告：為受試者 ID '{participant_id_raw}' 在 '{config.RAW_AUDIO_DIR}' 找到多個匹配檔案：{matching_files}。跳過此 ID。")
        else:
            logging.warning(f"警告：為受試者 ID '{participant_id_raw}' 在 '{config.RAW_AUDIO_DIR}' 找不到匹配的音訊檔案。跳過此 ID。")


    # 將匹配到的數據轉換為 DataFrame
    if final_metadata_list:
        metadata_df = pd.DataFrame(final_metadata_list)
        logging.info(f"成功匹配並載入 {len(metadata_df)} 筆有效資料 (音訊檔案與中繼資料匹配)。")

        # 檢查最終 DataFrame 是否包含所有內部必要的欄位
        # 注意：這裡我們不再需要 'score' 在必須欄位列表，因為我們不做回歸
        final_required_cols_check = [config.INTERNAL_FILENAME_COL, config.INTERNAL_ID_COL, config.INTERNAL_LABEL_COL, 'audio_full_path']
        if not all(col in metadata_df.columns for col in final_required_cols_check):
             logging.error(f"嚴重錯誤：最終處理用的中繼資料 DataFrame 構建失敗，缺少必要欄位。需要包含: {final_required_cols_check}")
             return None # 設為 None 停止後續處理
        else:
             logging.info(f"最終處理用的中繼資料 DataFrame 包含必要欄位: {final_required_cols_check}")
             logging.info("最終處理用的中繼資料前 5 行:")
             # 選擇性打印部分欄位以便閱讀
             print(metadata_df[[config.INTERNAL_FILENAME_COL, config.INTERNAL_ID_COL, config.INTERNAL_LABEL_COL, 'audio_full_path']].head().to_string())

             # 打印類別分佈
             logging.info(f"\n資料集類別分佈 ('{config.INTERNAL_LABEL_COL}' 欄位):\n{metadata_df[config.INTERNAL_LABEL_COL].value_counts().to_string()}")

             # 額外檢查：確認所有標籤是否在 config.LABEL_MAP 中 (如果要做分類訓練的話)
             unknown_labels = metadata_df[config.INTERNAL_LABEL_COL][~metadata_df[config.INTERNAL_LABEL_COL].isin(config.LABEL_MAP.keys())].unique().tolist()
             if unknown_labels:
                  logging.warning(f"警告： '{config.INTERNAL_LABEL_COL}' 欄位包含未定義的標籤值：{unknown_labels}。請檢查 config.LABEL_MAP 或您的原始數據。")

             # 確保 'score' 欄位存在 (即使都是 NaN)
             if config.INTERNAL_SCORE_COL not in metadata_df.columns:
                  metadata_df[config.INTERNAL_SCORE_COL] = np.nan
                  logging.info(f"已添加空的 '{config.INTERNAL_SCORE_COL}' 欄位。")

             return metadata_df # 成功構建 DataFrame，返回它

    else:
        logging.error("嚴重錯誤：未成功匹配任何音訊檔案和中繼資料。請檢查：")
        logging.error(f"1. 您的 metadata.csv 是否在 '{os.path.abspath(config.METADATA_FILE)}'？")
        logging.error(f"2. 您在 CSV 中指定的受試者編號 (欄位名 '{config.INPUT_ID_COL}') 是否與 '{os.path.abspath(config.RAW_AUDIO_DIR)}' 中的音訊檔案名匹配 (例如 '1' 對應 '0001_...' 或 '0001.wav')？")
        logging.error(f"3. 您的音訊檔案是否在 '{os.path.abspath(config.RAW_AUDIO_DIR)}'？")
        logging.error(f"4. 您的 CSV 欄位名稱是否確實是 '{config.INPUT_ID_COL}' 和 '{config.INPUT_LABEL_COL}'？")
        return None # 未匹配到數據，返回 None


# 如果直接運行此腳本，則執行測試載入
if __name__ == "__main__":
    # 可以在這裡添加一些 config 變數的檢查或設置，如果 config 需要在其他腳本中被初始化
    # 例如：os.environ['HF_TOKEN'] = '...'

    logging.info("--- 正在測試運行 metadata_processor.py ---")

    # 調用函數載入和匹配中繼資料
    matched_df = load_and_match_metadata()

    if matched_df is not None:
        logging.info("\nmetadata_processor.py 測試運行成功。最終匹配到的數據框如下：")
        print(matched_df.head().to_string())
        logging.info(f"總計 {len(matched_df)} 筆數據。")
    else:
        logging.error("\nmetadata_processor.py 測試運行失敗。無法載入或匹配中繼資料。")

    logging.info("--- metadata_processor.py 測試運行結束 ---")