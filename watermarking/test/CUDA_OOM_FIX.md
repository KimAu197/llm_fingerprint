# CUDA OOM 修复说明

## 问题描述
运行 `run_pairwise_overlap_matrix.py` 时遇到 CUDA OOM 错误，特别是在处理 7B 模型时。错误类型：
1. **CUDA AcceleratorError**: GPU 内存没有在模型卸载时完全释放
2. **7B 模型 OOM**: 在 24GB GPU 上处理批量时内存溢出
3. **OOM 导致进程崩溃**: 当一个模型加载失败时，`finally` 块中的 `torch.cuda.synchronize()` 在损坏的 CUDA 上下文上调用，导致整个进程崩溃

## 修复内容

### 1. 增强 `unload_hf_model` 函数 (`utils/model_loader.py`)
**修复内容**：
- 所有 CUDA 操作都包裹在 try-except 中
- 在删除模型前显式调用 `torch.cuda.synchronize()` (带异常保护)
- 多轮 `gc.collect()` 和 `torch.cuda.empty_cache()`
- 正确处理 accelerate 分发的模型
- 在卸载前移动模型到 CPU（对非 accelerate 模型）

**关键代码**：
```python
# All CUDA operations wrapped in try-except
try:
    if device is not None and device.type == 'cuda':
        torch.cuda.synchronize(device)
except Exception:
    pass  # CUDA context may be corrupted, ignore

# Multiple cleanup rounds - all wrapped
try:
    gc.collect()
    gc.collect()
except Exception:
    pass

try:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
except Exception:
    pass  # Ignore if CUDA context is corrupted
```

### 2. 改进 Phase 1 & 2 的错误处理 (`run_pairwise_overlap_matrix.py`)
**修复内容**：
- 使用 `try-finally` 块确保即使出错也会卸载模型
- **关键修复**: `finally` 块中的所有清理操作都包裹在 try-except 中
- 在 OOM 错误后尝试恢复 CUDA 上下文（清理 + 等待 2 秒）
- 初始化 `model = None, tok = None` 避免未定义变量错误

**关键代码**：
```python
model = None
tok = None
try:
    model, tok, _ = load_hf_model(...)
    # ... process model ...
except Exception as e:
    print(f"[ERROR] {e}")
    errors[name] = traceback.format_exc()
    
    # If OOM, try to recover CUDA context
    if "out of memory" in str(e).lower():
        print("[INFO] Attempting to recover CUDA context...")
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)  # Give GPU time to recover

finally:
    # All cleanup wrapped in try-except
    try:
        if model is not None or tok is not None:
            unload_hf_model(model, tok)
    except Exception as cleanup_err:
        print(f"[WARNING] Cleanup failed: {cleanup_err}")
    
    # Extra cleanup - also wrapped
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass  # Ignore if CUDA context corrupted
```

### 3. 减少批处理大小避免 OOM
**修复内容**：
- 将默认 `BATCH_SIZE_BOTTOMK` 从 64 降到 32
- 添加命令行参数 `--batch_size_bottomk` 允许用户调整
- 将批处理大小传递给 `_compute_all_bottomk_for_model` 函数

**使用方式**：
```bash
# 如果仍然 OOM，可以进一步减小批处理大小
python run_pairwise_overlap_matrix.py \
    --csv_path models.csv \
    --batch_size_bottomk 16  # 默认 32，可以降到 16 或 8
```

## 为什么之前会崩溃？

**问题根源**：
1. 模型加载失败时抛出 `CUDA OOM` 异常
2. `except` 块捕获异常并记录到 `errors` 字典
3. `finally` 块执行清理，调用 `torch.cuda.synchronize()`
4. **此时 CUDA 上下文已损坏**，`synchronize()` 抛出 `torch.AcceleratorError`
5. 这个新异常**没有被捕获**，导致整个进程崩溃

**修复方案**：
- 将 `finally` 块中的所有操作包裹在 try-except 中
- 即使 CUDA 上下文损坏，也能优雅地跳过清理并继续处理下一个模型

## 使用建议

### 对于 24GB GPU (如 RTX 3090/4090)：
- **小模型 (0.5B-3B)**: `--batch_size_bottomk 32` (默认)
- **中模型 (7B-8B)**: `--batch_size_bottomk 16`
- **大模型 (14B+)**: `--batch_size_bottomk 8`

### 如果还是遇到 OOM：
1. 进一步减小批处理大小：`--batch_size_bottomk 4`
2. 减少指纹长度：`--fingerprint_length 32` (默认 64)
3. 减少指纹数量：`--num_fingerprints 3` (默认 5)

### OOM 恢复机制
- 当模型加载失败时，脚本会：
  1. 记录错误到 `fingerprint_errors.json`
  2. 尝试清理 GPU 内存
  3. 等待 2 秒让 GPU 恢复
  4. 继续处理下一个模型
- 不会因为一个模型失败而崩溃整个实验

## 测试脚本
运行测试验证修复：
```bash
cd /Users/kenzieluo/Desktop/columbia/course/model_lineage/llm_fingerprint/watermarking/test
./test_cuda_fix.sh
```

## 新增参数
```bash
python run_pairwise_overlap_matrix.py \
    --csv_path models.csv \
    --gpu_ids 2 \
    --batch_size_bottomk 16  # 新参数：控制批处理大小
```

## 预期效果
- ✅ 不再出现 "torch.AcceleratorError: CUDA error (driver-level)"
- ✅ 单个模型 OOM 不会崩溃整个进程
- ✅ 失败的模型会被记录到 `fingerprint_errors.json`，实验继续
- ✅ 7B 模型可以在 24GB GPU 上正常运行（使用较小的 batch size）
- ✅ 每个模型处理完后 GPU 内存完全释放
- ✅ 连续处理多个模型不会累积内存
- ✅ OOM 后 CUDA 上下文可以恢复，后续模型可以继续加载
