# 這個程式碼片段可以放在 VS Code 的 Python 終端機中運行
# 或者創建一個新的 .py 檔案來運行

import numpy as np
import pandas as pd
import os
import logging # 導入 logging 來打印信息

# 從 config.py 導入設定（包括路徑）
try:
    import config
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # 確保日誌設定
    logging.info("config.py 載入成功。")
except ImportError:
    logging.error("錯誤：無法導入 config.py。請確保 config.py 存在。")
    exit() # 如果 config 無法載入，則退出

# 設定日誌級別以減少不必要的輸出 (可選)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# --- 指定要查看的 .npy 檔案 ---

# 您可以從 features_metadata.csv 檔案中找到一個 .npy 檔案的路徑
FEATURES_METADATA_FILE = config.FEATURES_METADATA_FILE # 從 config 中獲取路徑
ACOUSTIC_FEATURES_DIR = config.ACOUSTIC_FEATURES_DIR # 從 config 中獲取聲學特徵資料夾路徑

# 載入 features_metadata.csv 以獲取檔案列表和路徑
features_df = None
if os.path.exists(FEATURES_METADATA_FILE):
    try:
        features_df = pd.read_csv(FEATURES_METADATA_FILE)
        logging.info(f"成功載入特徵中繼資料檔案: {FEATURES_METADATA_FILE}")
        logging.info(f"共 {len(features_df)} 筆特徵記錄。")

        # 找到第一個成功保存的聲學特徵檔案路徑
        # 過濾掉 acoustic_feature_path 是 None 的記錄
        valid_features_df = features_df.dropna(subset=['acoustic_feature_path'])

        if not valid_features_df.empty:
            # 獲取第一個有效特徵檔案的完整路徑
            sample_feature_path = valid_features_df.iloc[0]['acoustic_feature_path']
            logging.info(f"找到第一個聲學特徵檔案樣本路徑: {sample_feature_path}")

            # 檢查檔案是否存在
            if os.path.exists(sample_feature_path):
                logging.info(f"\n--- 正在載入和查看檔案: {sample_feature_path} ---")
                try:
                    # 載入 .npy 檔案
                    feature_data = np.load(sample_feature_path)

                    logging.info(f"成功載入 .npy 檔案。")
                    logging.info(f"資料類型 (dtype): {feature_data.dtype}")
                    logging.info(f"維度 (shape): {feature_data.shape}")
                    # 期望 shape 應該是 [序列長度, 特徵維度] 例如 [x, 384] 或 [x, 768] 等

                    # 打印一部分數據 (例如前 5 行，前 10 個維度)
                    logging.info("\n數據樣本 (前 5 行，前 10 維):")
                    print(feature_data[:5, :10])

                    logging.info("\n--- 檔案查看結束 ---")

                except Exception as e:
                    logging.error(f"錯誤：無法載入或查看 .npy 檔案 {sample_feature_path}: {e}")

            else:
                 logging.error(f"錯誤：指定的聲學特徵檔案不存在: {sample_feature_path}")


        else:
            logging.warning("特徵中繼資料中沒有找到成功的聲學特徵記錄 (acoustic_feature_path 為 None)。")
            logging.warning("請檢查 whisper_feature_extractor.py 的運行日誌，看是否有聲學特徵提取失敗的錯誤。")


    except Exception as e:
        logging.error(f"錯誤：無法載入或解析特徵中繼資料檔案 {FEATURES_METADATA_FILE}: {e}")
        features_df = None
else:
    logging.warning(f"未找到特徵中繼資料檔案: {FEATURES_METADATA_FILE}")
    logging.warning("請先成功運行 whisper_feature_extractor.py 來生成此檔案。")