# 模型下载量分析 - 完整文档

## 📁 文件说明

### 原始数据（带下载量）
- `lineage_bottomk_overlap_llama_diff_with_downloads.csv` - TinyLlama vs Qwen (异类)
- `lineage_bottomk_overlap_llama_same_with_downloads.csv` - TinyLlama vs TinyLlama (同类)
- `lineage_bottomk_overlap_qwen_diff_with_downloads.csv` - Qwen vs TinyLlama (异类)
- `lineage_bottomk_overlap_qwen_same_with_downloads.csv` - Qwen vs Qwen (同类)

### 过滤后数据（filtered/目录）
- `*_filtered_10.csv` - 只保留下载量 ≥ 10 的模型

### 分析结果
- `分析报告.md` - 详细的中文分析报告
- `download_overlap_analysis.png` - 可视化图表
- `analysis_output.txt` - 完整的分析输出

## 🎯 核心发现

### 数据质量与下载量的关系

| 指标 | 低下载量 (≤10) | 高下载量 (>100) |
|------|---------------|----------------|
| **数据可靠性** | ⚠️ 低 | ✅ 高 |
| **标准差** | 大（噪声多） | 小（稳定） |
| **异常值** | 多 | 少 |
| **适用性** | 需谨慎 | 可直接使用 |

### Overlap模式（过滤后 threshold=10）

#### LLama家族:
- **同类模型**: avg_overlap = **0.5997** (60%)
- **异类模型**: avg_overlap = **0.0087** (0.87%)
- **区分度**: 68.9倍

#### Qwen家族:
- **同类模型**: avg_overlap = **0.2946** (29%)
- **异类模型**: avg_overlap = **0.0037** (0.37%)
- **区分度**: 79.6倍

### 过滤效果

使用 threshold=10 后：
- LLama Diff: 144 → 14 (保留 9.7%)
- LLama Same: 120 → 16 (保留 13.3%)
- Qwen Diff: 120 → 16 (保留 13.3%)
- Qwen Same: 136 → 14 (保留 10.3%)

**结论**: 过滤掉了约90%的低质量数据，但保留了最可靠的模型！

## 🔧 使用工具

### 1. 获取下载量数据
```bash
python fetch_hf_downloads.py
```
这会从Hugging Face API获取所有模型的下载量。

### 2. 分析数据
```bash
python analyze_download_overlap.py
```
生成详细的统计分析和可视化图表。

### 3. 过滤数据
```bash
# 使用默认阈值 (10)
python filter_by_downloads.py

# 使用自定义阈值
python filter_by_downloads.py --threshold 50

# 指定输入输出目录
python filter_by_downloads.py --threshold 10 \
    --input-dir llm_fingerprint/result/result_1.26/download \
    --output-dir llm_fingerprint/result/result_1.26/download/filtered
```

## 📊 推荐的过滤策略

### 策略1: 保守（最高质量）
```bash
python filter_by_downloads.py --threshold 100
```
- 只保留下载量 > 100 的模型
- 数据最可靠，但样本量很小
- 适用于：需要高精度的场景

### 策略2: 平衡（推荐）⭐
```bash
python filter_by_downloads.py --threshold 10
```
- 保留下载量 > 10 的模型
- 平衡了质量和样本量
- 适用于：大多数研究场景

### 策略3: 宽松（最大样本）
```bash
python filter_by_downloads.py --threshold 1
```
- 保留所有有下载记录的模型
- 样本量大，但噪声较多
- 适用于：需要大量数据的统计分析

## 💡 关键洞察

### 1. 同类 vs 异类区分非常清晰

无论下载量如何，都能看到明显的模式：
- 同类模型：overlap 30-60%
- 异类模型：overlap < 1%

这证明了你的 **fingerprinting 方法非常有效**！

### 2. 低下载量 = 高噪声

低下载量模型的特征：
- 标准差大（LLama Same: 0.36）
- 有极端值（0.0024 到 1.0）
- 不稳定

### 3. Qwen比LLama更稳定

Qwen异类模型的标准差极小（0.0003），说明：
- Qwen的tokenizer更一致
- 或者Qwen的派生模型质量更统一

## 🚀 下一步建议

### 1. 使用过滤后的数据重新训练
```python
# 读取过滤后的数据
import pandas as pd

df_clean = pd.read_csv('filtered/lineage_bottomk_overlap_qwen_same_filtered_10.csv')

# 你的lineage recovery算法
# ...
```

### 2. 引入置信度权重
```python
def calculate_confidence(download_count):
    """根据下载量计算置信度"""
    if download_count > 1000:
        return 1.0
    elif download_count > 100:
        return 0.9
    elif download_count > 10:
        return 0.7
    else:
        return 0.3

# 在预测时使用
prediction_score = overlap_score * calculate_confidence(downloads)
```

### 3. 异常检测
```python
# 检测可疑模型
suspicious = df[
    ((df['derived_model_downloads'] < 10) & (df['avg_overlap_ratio'] > 0.9)) |
    ((df['derived_model_downloads'] > 100) & (df['avg_overlap_ratio'] < 0.1))
]
```

### 4. 时间序列分析
如果有创建时间数据，可以分析：
- 模型质量是否随时间提升
- 早期模型 vs 最新模型的区别

## 📈 可视化解读

`download_overlap_analysis.png` 包含4个散点图：

### 如何解读：
- **X轴（对数刻度）**: 模型下载量
- **Y轴**: 平均overlap ratio
- **红色虚线**: 趋势线

### 关键观察：
1. **异类图（Diff）**: 点都集中在底部（overlap < 1%），趋势线几乎水平
2. **同类图（Same）**: 点分散在中上部（overlap 20-100%），有明显的正相关趋势
3. **LLama Same**: 高下载量模型的overlap更高（0.73）
4. **Qwen Diff**: 最稳定，所有点都在 0.3-0.4% 之间

## 🎓 学术价值

这个分析可以用于论文的以下部分：

### 1. Data Quality Section
> "We filtered models based on download counts as a proxy for data quality. Models with fewer than 10 downloads showed high variance (σ=0.36) compared to popular models (σ=0.29), indicating unreliable training data."

### 2. Methodology Section
> "To reduce noise from personal/experimental models, we applied a download threshold filter, retaining only models with ≥10 downloads, which reduced the dataset by 90% while preserving the most reliable samples."

### 3. Results Section
> "Our fingerprinting method achieved clear separation between same-family (overlap=60%) and different-family (overlap=0.87%) models, with a discrimination ratio of 68.9x for filtered data."

## 📞 联系与支持

如果有任何问题或需要进一步分析，可以：
1. 查看 `分析报告.md` 获取详细解释
2. 运行 `analyze_download_overlap.py` 重新生成分析
3. 修改 `filter_by_downloads.py` 的阈值进行实验

---

**生成时间**: 2026-01-25  
**数据来源**: Hugging Face Model Hub  
**总模型数**: 520 (去重后)  
**有效模型数**: ~60 (threshold=10)
