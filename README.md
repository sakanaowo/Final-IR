# Final-IR

## Chạy GPU trên Windows

Server tự chọn GPU nếu `torch.cuda.is_available()` trả về `True`. Nếu muốn ép dùng GPU:

```powershell
$env:MODEL_DEVICE="cuda"
uv run python server.py
```

Kiểm tra nhanh PyTorch có thấy GPU không:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Nếu kết quả là `False` hoặc `CPU only`, môi trường Windows đang cài torch bản CPU. Cài lại torch CUDA trước khi chạy `setup.py`/`server.py`.

Project này chỉ dùng text embedding/reranker nên không cần `torchvision` hoặc `torchaudio`. Nếu gặp lỗi `operator torchvision::nms does not exist`, gỡ `torchvision` vì bản đó đang lệch với `torch`.

```powershell
Remove-Item -Recurse -Force .venv
uv lock --upgrade-package torch
uv sync
```

Kiểm tra lại trước khi setup:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Nếu vẫn là `False`, kiểm tra driver NVIDIA:

```powershell
nvidia-smi
```

Sau đó chạy:

```powershell
uv run python setup.py
uv run python server.py
```

Khi server chạy đúng GPU, log sẽ hiện `Torch device: cuda` và `/health` sẽ có `model_device: "cuda"`, `torch_cuda_available: true`, `gpu_name`.
