# --- 導入必要的庫 ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

# --- 假設您的 config.py 已經定義了相關路徑和常量 ---
# 例如：
# import config # 如果您的 config.py 在 Python 路徑中
# 如果不在，您可能需要手動定義或添加到 sys.path
# 為了示例的獨立性，我們這裡手動定義一些常量
# 在您的實際使用中，請確保從 config.py 導入或正確設置這些值

# 假設的 config 值 (請替換為您 config.py 中的實際值)
FEATURES_METADATA_FILE = "processed_data/features_metadata.csv" # 您的 features_metadata.csv 路徑
INTERNAL_LABEL_COL = "label" # 您在 config 中定義的標籤欄位名
ACOUSTIC_FEATURE_PATH_COL = "acoustic_feature_path" # 您元數據中聲學特徵路徑的欄位名
# LABEL_MAP = {'Normal': 0, 'Dementia': 1} # 您在 config 中定義的標籤映射

# Whisper large-v2 Encoder 的 hidden size 通常是 1280
# 如果您用了其他模型或提取方式，請確認此值
# 您可以通過載入一個 .npy 文件並打印其 .shape[1] 來確認
# 例如：
# temp_feature = np.load(first_valid_npy_path)
# EXPECTED_HIDDEN_SIZE = temp_feature.shape[1]
EXPECTED_HIDDEN_SIZE = 1280 # 假設 Whisper large-v2 Encoder hidden size

# --- 1. 載入元數據 ---
try:
    features_df = pd.read_csv(FEATURES_METADATA_FILE)
    print(f"成功載入元數據: {FEATURES_METADATA_FILE}")
    print(f"總記錄數: {len(features_df)}")
    print(f"欄位: {features_df.columns.tolist()}")
except FileNotFoundError:
    print(f"錯誤: 元數據檔案 '{FEATURES_METADATA_FILE}' 未找到。請檢查路徑。")
    exit()
except Exception as e:
    print(f"錯誤: 載入元數據檔案時發生錯誤: {e}")
    exit()

# 檢查必要欄位是否存在
required_cols = [INTERNAL_LABEL_COL, ACOUSTIC_FEATURE_PATH_COL]
if not all(col in features_df.columns for col in required_cols):
    print(f"錯誤: 元數據檔案缺少必要欄位。需要: {required_cols}")
    exit()

# --- 2. 選擇和準備聲學特徵數據 ---
# 過濾掉聲學特徵路徑為空或檔案不存在的記錄
valid_features_df = features_df.dropna(subset=[ACOUSTIC_FEATURE_PATH_COL]).copy() # 使用 .copy()
valid_features_df = valid_features_df[valid_features_df[ACOUSTIC_FEATURE_PATH_COL].apply(lambda x: isinstance(x, str) and os.path.exists(x))]

if valid_features_df.empty:
    print("錯誤: 元數據中沒有找到有效的聲學特徵檔案路徑或檔案不存在。")
    exit()

print(f"找到 {len(valid_features_df)} 條包含有效聲學特徵路徑的記錄。")

# 為了演示，可以選擇一部分樣本，或者處理全部有效樣本 (如果計算資源允許)
# 這裡我們選擇所有有效的，但如果很多，t-SNE 會很慢
# sample_df = valid_features_df.sample(n=min(200, len(valid_features_df)), random_state=42)
sample_df = valid_features_df # 處理所有有效樣本

pooled_features_list = []
labels_for_analysis = []
filenames_for_analysis = [] # 可選，用於追蹤

print(f"\n正在處理和池化 {len(sample_df)} 個聲學特徵...")
for index, row in sample_df.iterrows():
    feature_path = row[ACOUSTIC_FEATURE_PATH_COL]
    label = row[INTERNAL_LABEL_COL]
    filename = row.get("filename", "N/A") # 假設元數據中有 'filename' 列

    try:
        feature_seq = np.load(feature_path)
        if feature_seq.ndim != 2 or feature_seq.shape[1] != EXPECTED_HIDDEN_SIZE:
            print(f"警告: 特徵 {filename} ({feature_path}) 維度異常 (shape: {feature_seq.shape})，期望 hidden_size={EXPECTED_HIDDEN_SIZE}。跳過此特徵。")
            continue

        # 平均池化 (Mean Pooling over sequence_length)
        pooled_feature = np.mean(feature_seq, axis=0) # Shape: (hidden_size,)
        pooled_features_list.append(pooled_feature)
        labels_for_analysis.append(label)
        filenames_for_analysis.append(filename)

    except Exception as e:
        print(f"錯誤: 處理特徵檔案 {feature_path} (來自 {filename}) 時發生錯誤: {e}")

if not pooled_features_list:
    print("錯誤: 未能成功池化任何聲學特徵。分析終止。")
    exit()

pooled_features_array = np.array(pooled_features_list)
labels_array = np.array(labels_for_analysis)

print(f"\n成功池化 {pooled_features_array.shape[0]} 個特徵。池化後特徵矩陣形狀: {pooled_features_array.shape}")

# --- 3. 特徵標準化 ---
scaler = StandardScaler()
scaled_pooled_features = scaler.fit_transform(pooled_features_array)
print("特徵已標準化。")

# --- 4. 使用 t-SNE 降維與可視化 ---
print("\n開始 t-SNE 降維與可視化...")
# 根據樣本數量調整 perplexity 和 n_iter
n_samples_for_tsne = scaled_pooled_features.shape[0]
perplexity_val = min(30, max(5, n_samples_for_tsne - 1)) # perplexity 應小於樣本數
n_iter_val = max(250, n_samples_for_tsne // 2) # 迭代次數可以多一些以確保收斂

if n_samples_for_tsne < 5: # t-SNE 至少需要幾個樣本
    print("t-SNE 樣本太少，跳過可視化。")
else:
    print(f"t-SNE 參數: n_samples={n_samples_for_tsne}, perplexity={perplexity_val}, n_iter={n_iter_val}")
    tsne = TSNE(n_components=2,
                random_state=42,
                perplexity=perplexity_val,
                n_iter=n_iter_val,
                learning_rate='auto', # 'auto' 通常是個好選擇 (200.0 for legacy, else (N/perplexity)/3)
                init='pca') # 'pca' 初始化通常更穩定
    
    features_2d = tsne.fit_transform(scaled_pooled_features)

    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels_array)
    # 為每個類別選擇不同的顏色和標記
    # colors = plt.cm.get_cmap('viridis', len(unique_labels)) # Matplotlib 3.7+
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))


    for i, label_val in enumerate(unique_labels):
        idx = (labels_array == label_val)
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1],
                    label=str(label_val), # 確保標籤是字符串
                    color=colors(i) if callable(colors) else colors[i], # 兼容不同 Matplotlib 版本
                    alpha=0.7, s=50) # s 是點的大小

    plt.title("t-SNE Visualization of Pooled Acoustic Features", fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)
    plt.legend(title="Label", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tsne_acoustic_features.png") # 保存圖像
    print("t-SNE 可視化圖像已保存為 tsne_acoustic_features.png")
    plt.show()


# --- 5. (可選) 訓練簡單的 Logistic Regression 模型並查看係數 ---
#    這部分需要您有 LABEL_MAP，並且標籤是文本形式需要轉換
#    假設 LABEL_MAP = {'Normal': 0, 'Dementia': 1} (需要從 config.py 獲取或在此定義)
LABEL_MAP = {'Normal': 0, 'Dementia': 1} # 示例，請確保與您的 config 一致

print("\n(可選) 嘗試訓練 Logistic Regression 模型...")
if len(np.unique(labels_array)) > 1: # 確保有多個類別
    # 將文本標籤轉為數字標籤
    try:
        numeric_labels = np.array([LABEL_MAP[lbl] for lbl in labels_array])
    except KeyError as e:
        print(f"錯誤: 標籤 '{e.args[0]}' 未在 LABEL_MAP 中定義。跳過 Logistic Regression。")
        print(f"元數據中的標籤有: {np.unique(labels_array)}")
        numeric_labels = None # 標記為處理失敗

    if numeric_labels is not None and len(np.unique(numeric_labels)) > 1:
        # 數據集劃分
        # 確保有足夠樣本進行劃分，並且每個類別在訓練集中都存在
        min_samples_per_class_train = 2 # 訓練集中每個類別至少需要的樣本數
        
        can_stratify = True
        for class_label in np.unique(numeric_labels):
            if np.sum(numeric_labels == class_label) < min_samples_per_class_train * 2: # 總樣本數至少是訓練集最小樣本數的兩倍
                can_stratify = False
                break
        
        if scaled_pooled_features.shape[0] > 5 and can_stratify : # 至少需要一些樣本
            X_train, X_test, y_train, y_test = train_test_split(
                scaled_pooled_features,
                numeric_labels,
                test_size=0.3,
                random_state=42,
                stratify=numeric_labels # 盡可能分層抽樣
            )

            if len(np.unique(y_train)) > 1: # 再次確認訓練集有多個類別
                log_reg = LogisticRegression(solver='liblinear', random_state=42, C=0.1, class_weight='balanced')
                log_reg.fit(X_train, y_train)
                accuracy = log_reg.score(X_test, y_test)
                print(f"Logistic Regression - Test Set Accuracy: {accuracy:.4f}")

                if hasattr(log_reg, "coef_") and log_reg.coef_.shape[0] == 1: # 二分類
                    coefficients = log_reg.coef_[0]
                    plt.figure(figsize=(15, 7))
                    
                    top_n = min(30, len(coefficients)) # 最多顯示30個
                    if len(coefficients) > top_n:
                        abs_coeffs = np.abs(coefficients)
                        top_indices = np.argsort(abs_coeffs)[-top_n:]
                        coeffs_to_plot = coefficients[top_indices]
                        x_labels = [str(i) for i in top_indices]
                        title_suffix = f"(Top {top_n} by Absolute Value)"
                    else:
                        coeffs_to_plot = coefficients
                        x_labels = [str(i) for i in range(len(coefficients))]
                        title_suffix = ""
                        
                    plt.bar(range(len(coeffs_to_plot)), coeffs_to_plot)
                    plt.xticks(range(len(coeffs_to_plot)), x_labels, rotation=45, ha="right")
                    plt.xlabel(f"Feature Dimension Index {title_suffix}", fontsize=12)
                    plt.ylabel("Coefficient Value", fontsize=12)
                    plt.title(f"Logistic Regression Coefficients for Pooled Acoustic Features", fontsize=14)
                    plt.grid(axis='y', linestyle='--')
                    plt.tight_layout()
                    plt.savefig("logistic_regression_coeffs_acoustic.png")
                    print("Logistic Regression 係數圖已保存為 logistic_regression_coeffs_acoustic.png")
                    plt.show()
            else:
                print("訓練集在劃分後只包含一個類別，無法訓練 Logistic Regression。")
        else:
            print("樣本數不足或類別不均衡，無法安全地劃分數據集並訓練 Logistic Regression。")
    else:
        if numeric_labels is not None: # numeric_labels 成功生成但只有一個類別
             print("數據集中只包含一個類別的有效標籤，無法訓練 Logistic Regression。")

else:
    print("數據集中只觀察到一個類別，無法進行分類任務或比較。")

print("\n聲學特徵分析腳本執行完畢。")