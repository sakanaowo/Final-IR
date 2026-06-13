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

Nếu kết quả là `False` hoặc `CPU only`, môi trường Windows đang cài torch bản CPU. Cài lại torch CUDA trước khi chạy `setup.py`/`server.py`:

```powershell
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Sau đó chạy lại:

```powershell
uv run python setup.py
uv run python server.py
```

Khi server chạy đúng GPU, log sẽ hiện `Torch device: cuda` và `/health` sẽ có `model_device: "cuda"`, `torch_cuda_available: true`, `gpu_name`.
