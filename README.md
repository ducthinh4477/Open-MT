# OpenMT Translation Demo

Demo dịch máy English -> Vietnamese dùng các Qwen model đã fine-tune với PhoMT trên Hugging Face:

- `dinhxuanhuy/Qwen3-0.6B-PhoMT-250K` là model mặc định, nhẹ hơn và phù hợp để demo ổn định.
- `dinhxuanhuy/Qwen2.5-3B-PhoMT-250k` được cấu hình load 4-bit để phù hợp GPU khoảng 8GB VRAM.

Ứng dụng gồm frontend React/Vite/Tailwind và backend FastAPI/Transformers. Backend lazy-load model khi gọi dịch lần đầu và chỉ giữ 1 model trên GPU tại một thời điểm.

## Yêu cầu

- Node.js `20.19+` hoặc `22.12+` cho frontend Vite.
- Python `3.10` hoặc `3.11` cho backend local.
- Docker + NVIDIA Container Toolkit nếu chạy GPU bằng Docker.

## Chạy local dev

Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Mở frontend tại `http://localhost:5173`.

## Chạy Docker

```bash
cp .env.example .env
docker compose up --build
```

Mở `http://localhost:3000`.

## Chạy trên server trường

```bash
git clone <repo-url>
cd Open-MT
cp .env.example .env
docker compose up --build
```

Server GPU cần cài NVIDIA driver và NVIDIA Container Toolkit. Compose đang bật `gpus: all` cho backend.

Kiểm tra GPU trong container:

```bash
docker exec -it <backend-container> nvidia-smi
```

## API

- `GET /api/health`: kiểm tra trạng thái, device và model hiện tại.
- `GET /api/models`: đọc danh sách model từ `backend/configs/models.yaml`.
- `POST /api/translate`: dịch English -> Vietnamese.

Ví dụ:

```bash
curl -X POST http://localhost:8000/api/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Good morning.",
    "source_lang": "en",
    "target_lang": "vi",
    "model_id": "qwen3_0_6b_phomt_250k",
    "max_new_tokens": 256,
    "temperature": 0
  }'
```

## Thêm model mới

Sửa `backend/configs/models.yaml` và thêm một mục mới:

```yaml
- id: your_model_id
  name: Your Model Name
  hf_id: org-or-user/model-name
  default: false
  load_in_4bit: false
  torch_dtype: float16
  max_input_chars: 5000
  description: Short note for the UI
```

Chỉ nên có một model mặc định, hoặc đặt `DEFAULT_MODEL_ID` trong `.env`.

## Lưu ý triển khai

- CPU fallback chạy được nhưng sẽ rất chậm, đặc biệt với model 3B.
- Model 3B nên dùng 4-bit. Lần chạy đầu sẽ tải model lâu và cần Internet.
- Nếu server không có Internet, cần preload Hugging Face cache hoặc mount volume model local vào `/models/huggingface`.
- Nếu model Hugging Face private, điền `HF_TOKEN` trong `.env`.

## Troubleshooting

- CUDA out of memory: chọn model 0.6B, giảm `max_new_tokens`, hoặc chuyển model để backend unload model cũ.
- Lỗi bitsandbytes: kiểm tra CUDA, NVIDIA driver và NVIDIA Container Toolkit.
- Frontend không gọi được backend: kiểm tra `VITE_API_BASE_URL`, CORS và port `8000`.
- Model tải chậm: giữ volume `hf_cache` để cache không bị mất giữa các lần chạy.
