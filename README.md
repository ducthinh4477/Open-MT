# OpenMT Translation Demo

Demo dịch máy English <-> Vietnamese dùng các Qwen model đã fine-tune với PhoMT trên Hugging Face và các model local trong thư mục `Multiple files`:

- `dinhxuanhuy/Qwen2.5-3B-PhoMT-500kMulti` là model mặc định, dùng revision retrain `e1db5b2316aa8c684bd06b95910a3c75fdf13858`.
- `dinhxuanhuy/Qwen3-0.6B-PhoMT-250K` nhẹ hơn và phù hợp để demo ổn định khi cần.
- `dinhxuanhuy/Qwen2.5-3B-PhoMT-250k` được cấu hình load 4-bit để phù hợp GPU khoảng 8GB VRAM.
- `dinhxuanhuy/llama-3.2-1B-PhoMT-250k` có trong danh sách model để chọn từ UI.

Ứng dụng gồm frontend React/Vite/Tailwind và backend FastAPI/Transformers. Backend lazy-load model khi gọi dịch lần đầu và chỉ giữ 1 model trên GPU tại một thời điểm.

Quy ước hướng dịch:

- Riêng `dinhxuanhuy/Qwen2.5-3B-PhoMT-500kMulti` tự phát hiện tiếng Việt/tiếng Anh và dịch sang ngôn ngữ còn lại.
- Các model còn lại được khóa cứng English -> Vietnamese.
- Attention map được trả về sau mỗi lần dịch khi model có attention khả dụng. Qwen/Llama hiển thị self-attention của token dịch về token input; local Transformer hiển thị cross-attention decoder-encoder. Local BiLSTM không có attention layer thật nên không hiển thị map.

## Thành viên nhóm

1. Vũ Minh Đức - 23110094
2. Đinh Xuân Huy - 23110102
3. Trần Huỳnh Chí Nguyên - 23110136
4. Phùng Lê Thanh Quân - 23110145
5. Nguyễn Đức Thịnh - 23110156

## Yêu cầu

- Node.js `20.19+` hoặc `22.12+` cho frontend Vite.
- Python `3.10` hoặc `3.11` cho backend local.
- Podman/Podman Compose cho frontend container.
- Python virtualenv trong `backend/.venv` cho backend FastAPI.
- `cloudflared` nếu cần public URL qua Cloudflare Quick Tunnel.

## Chạy demo hiện tại trên server

Chạy từ đầu theo thứ tự dưới đây.

### 1. Chạy frontend container

```bash
cd /home/sysadmin/Open-MT
podman-compose up --build -d
```

Frontend chạy ở `http://127.0.0.1:3000`. Container dùng `network_mode: host` và proxy `/api` về backend ở `127.0.0.1:8000`.

### 2. Chạy backend trong tmux

```bash
tmux new -s openmt-backend
cd /home/sysadmin/Open-MT/backend
source .venv/bin/activate
export HF_HOME=/home/sysadmin/.cache/huggingface
export TRANSFORMERS_CACHE=/home/sysadmin/.cache/huggingface
export DEFAULT_MODEL_ID=qwen2_5_3b_phomt_500k_multi
export MODEL_CACHE_SIZE=1
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Nếu muốn chuyển qua lại giữa các model mượt hơn và GPU còn đủ VRAM, có thể thử `MODEL_CACHE_SIZE=2` để backend giữ thêm model đã warm trong bộ nhớ. Với GPU 8GB, nên giữ `1` nếu thường chuyển giữa các model 3B để tránh CUDA out of memory.

Detach khỏi tmux bằng `Ctrl-b` rồi `d`. Vào lại session backend:

```bash
tmux attach -t openmt-backend
```

Kiểm tra backend:

```bash
curl http://127.0.0.1:8000/api/health
```

### 3. Public demo tạm thời qua Cloudflare Quick Tunnel

Mở thêm một tmux session riêng cho Cloudflare:

```bash
tmux new -s openmt-cloudflare
cd /home/sysadmin/Open-MT
/home/sysadmin/bin/cloudflared tunnel --url http://localhost:3000 --no-autoupdate
```

Nếu shell đã có `cloudflared` trong `PATH`, có thể chạy ngắn hơn:

```bash
cloudflared tunnel --url http://localhost:3000 --no-autoupdate
```

Sau khi chạy, Cloudflare sẽ in ra URL dạng `https://...trycloudflare.com`. Dùng URL đó để mở demo public. Giữ session `openmt-cloudflare` chạy trong tmux; detach bằng `Ctrl-b` rồi `d`.

Vào lại session Cloudflare:

```bash
tmux attach -t openmt-cloudflare
```

### 4. Gắn domain riêng Namecheap vĩnh viễn

Quick Tunnel ở trên chỉ phù hợp để demo nhanh vì URL `trycloudflare.com` có thể đổi. Muốn dùng domain riêng lâu dài, dùng Cloudflare Tunnel dạng named tunnel và chạy `cloudflared` bằng systemd.

#### 4.1. Trỏ domain Namecheap sang Cloudflare

Trong Cloudflare:

1. Add site/domain của bạn vào Cloudflare.
2. Cloudflare sẽ cấp 2 nameserver, ví dụ `alice.ns.cloudflare.com` và `bob.ns.cloudflare.com`.

Trong Namecheap:

1. Vào `Domain List`.
2. Chọn domain cần dùng.
3. Chọn `Manage`.
4. Ở mục `Nameservers`, chọn `Custom DNS`.
5. Nhập 2 nameserver Cloudflare cấp.
6. Lưu lại.

Lưu ý: nếu domain đang dùng email hoặc dịch vụ khác, hãy copy các record `MX`, `TXT`, `A`, `CNAME` hiện có sang Cloudflare DNS trước khi đổi nameserver để tránh mất mail/web cũ. DNS có thể cần vài giờ, đôi khi tới 24-48 giờ, để propagate đầy đủ.

#### 4.2. Tạo named tunnel cho OpenMT

Ví dụ dùng subdomain `openmt.example.com`. Thay `openmt.example.com` bằng domain/subdomain thật của bạn.

```bash
cd /home/sysadmin/Open-MT

# Mở browser để login Cloudflare và chọn domain.
/home/sysadmin/bin/cloudflared tunnel login

# Tạo tunnel cố định tên openmt.
/home/sysadmin/bin/cloudflared tunnel create openmt

# Xem tunnel id vừa tạo.
/home/sysadmin/bin/cloudflared tunnel list
```

Sau khi tạo tunnel, ghi lại `Tunnel ID` được in ra. Tạo file config:

```bash
mkdir -p /home/sysadmin/.cloudflared
nano /home/sysadmin/.cloudflared/config.yml
```

Nội dung mẫu, thay `<TUNNEL_ID>` và `openmt.example.com`:

```yaml
tunnel: <TUNNEL_ID>
credentials-file: /home/sysadmin/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: openmt.example.com
    service: http://localhost:3000
  - service: http_status:404
```

Tạo DNS record trỏ domain vào tunnel:

```bash
/home/sysadmin/bin/cloudflared tunnel route dns openmt openmt.example.com
```

Chạy thử:

```bash
/home/sysadmin/bin/cloudflared tunnel run openmt
```

Nếu mở `https://openmt.example.com` thấy UI OpenMT thì dừng lệnh chạy thử bằng `Ctrl-c`, rồi cài chạy nền bằng systemd.

#### 4.3. Cài cloudflared thành service để chạy vĩnh viễn

```bash
sudo /home/sysadmin/bin/cloudflared \
  --config /home/sysadmin/.cloudflared/config.yml \
  service install \
  --no-update-service

sudo systemctl enable --now cloudflared
sudo systemctl status cloudflared
```

Sau bước này, tunnel sẽ tự chạy lại sau reboot. Không cần giữ tmux cho Cloudflare nữa.

Kiểm tra log khi cần:

```bash
journalctl -u cloudflared -f
```

Nếu muốn dùng domain gốc thay vì subdomain, đổi `openmt.example.com` thành `example.com` ở cả `hostname` trong `config.yml` và lệnh `tunnel route dns`. Khuyến nghị dùng subdomain như `openmt.example.com` để không ảnh hưởng website/email đang có ở domain gốc.

### 5. Kiểm tra nhanh toàn bộ demo

```bash
curl http://127.0.0.1:3000
curl http://127.0.0.1:8000/api/models
curl -X POST http://127.0.0.1:8000/api/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Good morning.",
    "source_lang": "en",
    "target_lang": "vi",
    "model_id": "qwen2_5_3b_phomt_500k_multi",
    "max_new_tokens": 256,
    "temperature": 1,
    "use_beam_search": true
  }'
```

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



## API

- `GET /api/health`: kiểm tra trạng thái, device và model hiện tại.
- `GET /api/models`: đọc danh sách model từ `backend/configs/models.yaml`.
- `POST /api/models/{model_id}/warm`: nạp model trước khi dịch, giúp UI chuyển model mượt hơn.
- `POST /api/translate`: dịch English <-> Vietnamese, tùy `source_lang` và `target_lang`.
- `use_beam_search`: chỉ áp dụng cho model local không phải LLM; Qwen/Llama bỏ qua cờ này.
- `attention_map`: ma trận attention token-level nếu model hỗ trợ.

Ví dụ:

```bash
curl -X POST http://localhost:8000/api/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Good morning.",
    "source_lang": "en",
    "target_lang": "vi",
    "model_id": "qwen2_5_3b_phomt_500k_multi",
    "max_new_tokens": 256,
    "temperature": 1,
    "use_beam_search": true
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
