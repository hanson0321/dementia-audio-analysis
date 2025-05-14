# config.py

# --- 標準庫導入 ---
import os
import logging
import sys # 用於退出程式或檢查模組

# --- 第三方庫導入 (確保已在您的 conda 環境中安裝) ---
# 請在您的 conda 環境中運行以下指令來安裝所有必要的套件：
# conda install -c conda-forge numpy pandas torch torchaudio librosa soundfile tqdm matplotlib scikit-learn -y
# pip install transformers huggingface_hub datasets accelerate tokenizers safetensors openai-whisper audiomentations captum

import torch
# torch 和 torchaudio 會在 audio_utils 中導入，這裡只是檢查 cuda
# import torchaudio
# import librosa
# import soundfile as sf
# from tqdm.auto import tqdm
# from scipy.io import wavfile

# Hugging Face 生態系統導入
try:
    # 從 transformers 導入一些基礎的模組，後續在 feature extractor 中會導入更多
    from transformers import AutoConfig
    from huggingface_hub import login, HfFolder # 用于可能的认证或token管理
    # from datasets import Dataset, Audio # 如果後續需要構建 Hugging Face Dataset
    print("Hugging Face Hub 和 Transformers 基礎模組導入嘗試成功。")
except ImportError as e:
    print(f"--- 嚴重錯誤：無法導入 Hugging Face Hub 或 Transformers 基礎模組 ---")
    print(f"詳細錯誤：{e}")
    print("\n請確認您已在conda環境中並成功安裝套件：")
    print("conda activate your_dementia_env") # 替換為您的環境名稱
    print("pip install transformers huggingface_hub datasets accelerate tokenizers safetensors openai-whisper audiomentations captum")
    # 在 config 階段如果核心庫導入失敗，直接退出
    sys.exit("無法繼續，請解決套件導入問題。")

# 導入資料增強庫 (稍後在 audio_utils 中初始化和使用)
try:
    import audiomentations as audioaug
    print("Audiomentations 導入成功。")
    AUDIO_AUGMENTATION_AVAILABLE = True
except ImportError as e:
     print(f"警告：無法導入 Audiomentations 庫: {e}")
     print("資料增強功能將不可用。請確認已安裝：pip install audiomentations")
     AUDIO_AUGMENTATION_AVAILABLE = False


# 導入 Captum (模型解釋) (稍後在 model_trainer 或單獨腳本中使用)
try:
    import captum
    print("Captum 導入成功。")
    CAPTUM_AVAILABLE = True
except ImportError as e:
     print(f"警告：無法導入 Captum 庫: {e}")
     print("模型解釋功能將不可用。請確認已安裝：pip install captum")
     CAPTUM_AVAILABLE = False


# --- 設定日誌 ---
# 清除已有的 handlers 以避免重複輸出日誌 (在腳本多次運行時可能有用)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# 設定基礎日誌格式和級別
# 根據您的需求調整 level，例如 logging.INFO, logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 進一步細化某些庫的日誌級別，減少不必要的輸出
logging.getLogger("transformers").setLevel(logging.ERROR) # 只顯示錯誤
logging.getLogger("huggingface_hub").setLevel(logging.WARNING) # 顯示警告及以上
# torchaudio 和 audiomentations 的日誌級別在它們自己的模塊中設定可能更好，或者在這裡設定
# logging.getLogger("torchaudio").setLevel(logging.WARNING)
# if AUDIO_AUGMENTATION_AVAILABLE:
#     logging.getLogger("audiomentations").setLevel(logging.WARNING)

logging.info("基礎套件導入和日誌設定完成。")


# --- 設定設備 ---
# 在 VS Code 的 Conda 環境中，確保您有正確的 PyTorch 版本 (cpu 或 cuda)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"將使用設備: {DEVICE}")
if torch.cuda.is_available():
    logging.info(f"PyTorch 版本: {torch.__version__}")
    # torchaudio 版本在 audio_utils 中打印，因為那裡才強制導入 torchaudio
    # logging.info(f"Torchaudio 版本: {torchaudio.__version__}")
    logging.info(f"CUDA 可用，使用設備: {torch.cuda.get_device_name(0)}")
else:
     logging.info(f"PyTorch 版本: {torch.__version__}")
     # logging.info(f"Torchaudio 版本: {torchaudio.__version__}")
     logging.info("CUDA 不可用，將使用 CPU。")


# --- 設置 Hugging Face Token ---
# 從環境變數讀取或通過 login() 交互式登錄
# 推薦方式是設置 HF_TOKEN 環境變數 (.env 文件或系統設置)
# 如果您使用 .env 文件，請確保在您的主腳本 (run_pipeline.py) 開頭取消註解下面兩行並確保已安裝 python-dotenv
# from dotenv import load_dotenv
# load_dotenv() # 這會在調用它的腳本中加載 .env

try:
    # HfFolder.get_token() 會嘗試從環境變數 HF_TOKEN 或本地緩存文件獲取 token
    token = HfFolder.get_token()
    if token:
        # logging.info("Hugging Face Token 已找到 (來自緩存或環境變數)。") # 在 config 階段就不需要打印這個 info 了
        # 可以在需要的時候在其他模塊中調用 login(token=token) 來明確登錄
        pass # token 找到了就好
    else:
         logging.warning("未找到 Hugging Face Token。部分需要驗證的模型下載可能會失敗。")

except Exception as e:
    logging.warning(f"Hugging Face Token 檢查失敗: {e}")
    logging.warning("部分需要驗證的模型下載可能會失敗。請確保已設置 HF_TOKEN 環境變數或通過 login() 成功登錄。")


# --- 定義資料路徑與設定 ---

# 獲取腳本所在的目錄作為專案根目錄 (這個 config.py 應該和 run_pipeline.py 在同一目錄)
# 假設專案根目錄是包含 config.py 的目錄
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
logging.info(f"專案根目錄被偵測為: {PROJECT_ROOT}")

# 將 metadata.csv 放在專案根目錄下的 data 子資料夾
DATA_BASE_DIR = os.path.join(PROJECT_ROOT, "data")
METADATA_FILE = os.path.join(DATA_BASE_DIR, "metadata.csv")

# 根據您的最新情況，原始語音檔案放在 project_root/wav_data 下
RAW_AUDIO_DIR = os.path.join(PROJECT_ROOT, "wav_data") # 原始 wav 檔案存放處

# 預處理後資料存放路徑 (保持放在專案根目錄下的子資料夾)
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")

# 各個處理步驟的輸出子資料夾
CHUNKS_DIR = os.path.join(PROCESSED_DATA_DIR, "chunks") # 切片語音存放處
AUGMENTED_DIR = os.path.join(PROCESSED_DATA_DIR, "augmented") # 增強語音存放處
ACOUSTIC_FEATURES_DIR = os.path.join(PROCESSED_DATA_DIR, "acoustic_features") # 聲學特徵 (.npy) 存放處
LINGUISTIC_FEATURES_DIR = os.path.join(PROCESSED_DATA_DIR, "linguistic_features") # 語言特徵 (.npy) 存放處
# DIARIZED_DIR = os.path.join(PROCESSED_DATA_DIR, "diarized") # 語者分離後存放處 (暫時跳過)

# 各個處理步驟生成的中繼資料檔案
CHUNK_METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, "chunk_metadata.csv") # 語音切片後的中繼資料
FEATURES_METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, "features_metadata.csv") # 特徵提取後 (包含轉錄) 的中繼資料


# 語音參數設定
SAMPLE_RATE = 16000 # 16kHz
N_CHANNELS = 1 # 單聲道
AUDIO_DTYPE = torch.float32 # 音訊載入時的目標 dtype

# 語音處理參數
CHUNK_DURATION_SEC = 30 # 語音切片長度 (秒) - Whisper 輸入限制
# CHUNK_SAMPLE_SIZE 在 audio_utils 中根據 SAMPLE_RATE 和 CHUNK_DURATION_SEC 計算

# Hugging Face 模型名稱
WHISPER_MODEL_NAME = "openai/whisper-large-v2" # 或 base, medium, large 等
# 請根據您的語料語言選擇 BERT 模型：
BERT_MODEL_NAME = "bert-base-chinese" # 假設您的語料是中文，請修改；英文用 bert-base-uncased；多語言用 bert-base-multilingual-cased

# 定義您的原始 CSV 檔案中的實際欄位名稱 - <<<< 請確保這裡與您的 metadata.csv 欄位名稱完全一致
INPUT_ID_COL = '受試者編號'
INPUT_LABEL_COL = '分析結果(分類)'

# 定義程式內部使用的標準欄位名稱
INTERNAL_FILENAME_COL = 'filename' # 存儲匹配到的 wav 檔案名 (例如 0001_姓名.wav)
INTERNAL_ID_COL = 'participant_id' # 存儲 CSV 中的原始受試者編號 (例如 1)
INTERNAL_LABEL_COL = 'label' # 存儲 CSV 中的原始分析結果 (例如 Normal, Dementia)
INTERNAL_SCORE_COL = 'score' # 程式內部使用這個名稱，即使 CSV 中沒有數據，仍然保留作為 placeholder，值都是 NaN

# 分類標籤的映射（將文字標籤 Normal/Dementia 轉換為數字）
# 如果您的分類標籤是其他值，請修改這裡
LABEL_MAP = {'Normal': 0, 'Dementia': 1}
# 確保 CSV 中的標籤值是 LABEL_MAP 中定義的 Key，否則需要清洗數據 (例如處理 Inconclusive)


logging.info("\n全局設定完成。請檢查資料路徑是否正確。")
logging.info(f"專案根目錄: {PROJECT_ROOT}")
logging.info(f"期望原始語音路徑 (語音檔案所在目錄): {os.path.abspath(RAW_AUDIO_DIR)}")
logging.info(f"期望中繼資料路徑 (metadata.csv 所在路徑): {os.path.abspath(METADATA_FILE)}")
logging.info(f"處理後數據基礎目錄: {os.path.abspath(PROCESSED_DATA_DIR)}")
logging.info(f"語音切片存儲於: {os.path.abspath(CHUNKS_DIR)}")
logging.info(f"增強語音存儲於: {os.path.abspath(AUGMENTED_DIR)}")
logging.info(f"聲學特徵存儲於: {os.path.abspath(ACOUSTIC_FEATURES_DIR)}")
logging.info(f"語言特徵存儲於: {os.path.abspath(LINGUISTIC_FEATURES_DIR)}")
logging.info(f"語音切片中繼資料: {os.path.abspath(CHUNK_METADATA_FILE)}")
logging.info(f"特徵提取後中繼資料: {os.path.abspath(FEATURES_METADATA_FILE)}")

# 確保所有輸出資料夾存在 (在 config 中創建一次)
os.makedirs(DATA_BASE_DIR, exist_ok=True) # 確保 data 資料夾存在
os.makedirs(RAW_AUDIO_DIR, exist_ok=True) # 確保 wav_data 資料夾存在
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(AUGMENTED_DIR, exist_ok=True)
os.makedirs(ACOUSTIC_FEATURES_DIR, exist_ok=True)
os.makedirs(LINGUISTIC_FEATURES_DIR, exist_ok=True)

logging.info("\n--- config.py 載入完成 ---")