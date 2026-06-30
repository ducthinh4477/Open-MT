from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_BEAM_SIZE = 4
DEFAULT_LENGTH_PENALTY = 0.6
LOCAL_MAX_NEW_TOKENS = 128


def clean_and_tokenize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'([.,!?"():;])', r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_beam_score(score: float, token_count: int, length_penalty: float) -> float:
    length = max(1, token_count)
    penalty = ((5 + length) / 6) ** length_penalty
    return score / penalty


class VocabOnlySentencePieceTokenizer:
    def __init__(self, vocab_path: Path) -> None:
        self.id_to_piece_map: list[str] = []
        with vocab_path.open("r", encoding="utf-8") as file:
            for line in file:
                piece = line.rstrip("\n").split("\t", 1)[0]
                if piece:
                    self.id_to_piece_map.append(piece)

        self.piece_to_id_map = {
            piece: idx for idx, piece in enumerate(self.id_to_piece_map)
        }
        self.unk_id = self.piece_to_id_map.get("<unk>", 3)
        self.special_ids = {
            self.piece_to_id_map[piece]
            for piece in ("<pad>", "<sos>", "<eos>", "<unk>")
            if piece in self.piece_to_id_map
        }
        self.max_piece_len = max((len(piece) for piece in self.id_to_piece_map), default=1)

    def piece_to_id(self, piece: str) -> int:
        return self.piece_to_id_map.get(piece, self.unk_id)

    def id_to_piece(self, token_id: int) -> str:
        if 0 <= token_id < len(self.id_to_piece_map):
            return self.id_to_piece_map[token_id]
        return "<unk>"

    def get_piece_size(self) -> int:
        return len(self.id_to_piece_map)

    def encode(self, text: str, out_type: type = int) -> list[int] | list[str]:
        normalized = "▁" + text.strip().replace(" ", "▁")
        pieces: list[str] = []
        index = 0
        while index < len(normalized):
            matched_piece = ""
            max_end = min(len(normalized), index + self.max_piece_len)
            for end in range(max_end, index, -1):
                candidate = normalized[index:end]
                if candidate in self.piece_to_id_map:
                    matched_piece = candidate
                    break
            if matched_piece:
                pieces.append(matched_piece)
                index += len(matched_piece)
            else:
                pieces.append("<unk>")
                index += 1

        if out_type is str:
            return pieces
        return [self.piece_to_id(piece) for piece in pieces]

    def decode(self, token_ids: list[int]) -> str:
        pieces = [
            self.id_to_piece(token_id)
            for token_id in token_ids
            if token_id not in self.special_ids
        ]
        return "".join(pieces).replace("▁", " ").strip()


def load_tokenizer(tokenizer_path: Path) -> Any:
    if tokenizer_path.suffix == ".vocab":
        return VocabOnlySentencePieceTokenizer(tokenizer_path)

    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=str(tokenizer_path))


def load_transformer_state(weight_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(weight_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def infer_transformer_config(state: dict[str, torch.Tensor]) -> dict[str, int]:
    vocab_size, d_model = state["src_embedding.weight"].shape
    num_layers = len(
        {
            key.split(".")[3]
            for key in state
            if key.startswith("transformer.encoder.layers.")
        }
    )
    dim_feedforward = state["transformer.encoder.layers.0.linear1.weight"].shape[0]
    nhead = 8 if d_model % 8 == 0 else 4
    return {
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": int(dim_feedforward),
    }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.pe[: value.size(0)]
        return self.dropout(value)


class BPETransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.trg_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_emb)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        tgt_emb = self.pos_decoder(self.trg_embedding(tgt) * math.sqrt(self.d_model))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(
            tgt.device
        )
        return self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)


class LocalTransformerTranslator:
    def __init__(
        self,
        weight_path: Path,
        tokenizer_path: Path,
        device_name: str,
    ) -> None:
        self.device = torch.device(device_name)
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.pad_id = self.tokenizer.piece_to_id("<pad>")
        self.sos_id = self.tokenizer.piece_to_id("<sos>")
        self.eos_id = self.tokenizer.piece_to_id("<eos>")

        state = load_transformer_state(weight_path)
        self.model = BPETransformer(**infer_transformer_config(state))
        if state["src_embedding.weight"].shape[0] != self.tokenizer.get_piece_size():
            raise ValueError(
                f"Tokenizer vocab size {self.tokenizer.get_piece_size()} does not match "
                f"checkpoint vocab size {state['src_embedding.weight'].shape[0]}."
            )
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
        temperature: float,
        use_beam_search: bool = True,
    ) -> str:
        if source_lang != "en" or target_lang != "vi":
            raise ValueError("This local Transformer only supports English to Vietnamese.")

        cleaned = clean_and_tokenize_text(text)
        token_ids = [self.sos_id, *self.tokenizer.encode(cleaned, out_type=int), self.eos_id]
        src = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(1)

        with torch.inference_mode():
            memory = self.model.encode(src)
            max_len = min(max_new_tokens, LOCAL_MAX_NEW_TOKENS)
            if use_beam_search:
                output_ids = self._beam_search(
                    memory,
                    max_len=max_len,
                    beam_size=DEFAULT_BEAM_SIZE,
                    length_penalty=DEFAULT_LENGTH_PENALTY,
                )
            else:
                output_ids = self._greedy_decode(memory, max_len=max_len)

        return self._decode(output_ids)

    def translate_with_attention(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
        temperature: float,
        use_beam_search: bool = True,
    ) -> tuple[str, dict[str, Any] | None]:
        if source_lang != "en" or target_lang != "vi":
            raise ValueError("This local Transformer only supports English to Vietnamese.")

        cleaned = clean_and_tokenize_text(text)
        source_content_ids = self.tokenizer.encode(cleaned, out_type=int)
        token_ids = [self.sos_id, *source_content_ids, self.eos_id]
        src = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(1)

        with torch.inference_mode():
            memory = self.model.encode(src)
            max_len = min(max_new_tokens, LOCAL_MAX_NEW_TOKENS)
            if use_beam_search:
                output_ids = self._beam_search(
                    memory,
                    max_len=max_len,
                    beam_size=DEFAULT_BEAM_SIZE,
                    length_penalty=DEFAULT_LENGTH_PENALTY,
                )
            else:
                output_ids = self._greedy_decode(memory, max_len=max_len)

            attention_map = self._build_attention_map(
                source_content_ids,
                output_ids,
                memory,
            )

        return self._decode(output_ids), attention_map

    def _greedy_decode(self, memory: torch.Tensor, max_len: int) -> list[int]:
        output_ids: list[int] = []
        token_ids = [self.sos_id]

        for _ in range(max_len):
            tgt = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(1)
            decoded = self.model.decode(tgt, memory)
            logits = self.model.fc_out(decoded[-1, 0])
            next_id = int(logits.argmax(dim=-1).item())
            if next_id == self.eos_id:
                break
            output_ids.append(next_id)
            token_ids.append(next_id)

        return output_ids

    def _beam_search(
        self,
        memory: torch.Tensor,
        max_len: int,
        beam_size: int,
        length_penalty: float,
    ) -> list[int]:
        beams: list[tuple[list[int], float, bool]] = [([self.sos_id], 0.0, False)]

        for _ in range(max_len):
            candidates: list[tuple[list[int], float, bool]] = []

            for token_ids, score, is_finished in beams:
                if is_finished:
                    candidates.append((token_ids, score, is_finished))
                    continue

                tgt = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(1)
                decoded = self.model.decode(tgt, memory)
                logits = self.model.fc_out(decoded[-1, 0])
                log_probs = torch.log_softmax(logits, dim=-1)
                top_scores, top_ids = torch.topk(log_probs, beam_size)

                for token_score, token_id in zip(top_scores.tolist(), top_ids.tolist()):
                    next_id = int(token_id)
                    candidates.append(
                        (
                            [*token_ids, next_id],
                            score + float(token_score),
                            next_id == self.eos_id,
                        )
                    )

            beams = sorted(
                candidates,
                key=lambda beam: normalize_beam_score(
                    beam[1],
                    len(beam[0]) - 1,
                    length_penalty,
                ),
                reverse=True,
            )[:beam_size]
            if all(is_finished for _, _, is_finished in beams):
                break

        best_tokens = max(
            beams,
            key=lambda beam: normalize_beam_score(
                beam[1],
                len(beam[0]) - 1,
                length_penalty,
            ),
        )[0]
        return [token_id for token_id in best_tokens[1:] if token_id != self.eos_id]

    def _decode(self, token_ids: list[int]) -> str:
        special_ids = {self.pad_id, self.sos_id, self.eos_id}
        clean_ids = [token_id for token_id in token_ids if token_id not in special_ids]
        return self.tokenizer.decode(clean_ids).strip()

    def _build_attention_map(
        self,
        source_ids: list[int],
        output_ids: list[int],
        memory: torch.Tensor,
    ) -> dict[str, Any] | None:
        clean_output_ids = [
            token_id
            for token_id in output_ids
            if token_id not in {self.pad_id, self.sos_id, self.eos_id}
        ]
        if not source_ids or not clean_output_ids:
            return None

        decoder_ids = [self.sos_id, *clean_output_ids[:-1]]
        tgt = torch.tensor(decoder_ids, dtype=torch.long, device=self.device).unsqueeze(1)
        _, cross_attention = self._decode_with_cross_attention(tgt, memory)
        if cross_attention is None:
            return None

        weights = cross_attention[: len(clean_output_ids), 1 : 1 + len(source_ids)]
        weights = weights[:36, :36]
        source_ids = source_ids[: weights.shape[1]]
        clean_output_ids = clean_output_ids[: weights.shape[0]]
        matrix = [
            self._normalize_attention_row(row)
            for row in weights.detach().float().cpu().tolist()
        ]
        return {
            "source_tokens": [self._token_label(token_id) for token_id in source_ids],
            "target_tokens": [self._token_label(token_id) for token_id in clean_output_ids],
            "weights": matrix,
        }

    def _decode_with_cross_attention(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.model.pos_decoder(
            self.model.trg_embedding(tgt) * math.sqrt(self.model.d_model)
        )
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(
            tgt.device
        )
        cross_attention = None

        for layer in self.model.transformer.decoder.layers:
            if getattr(layer, "norm_first", False):
                x = x + self._self_attention_block(layer, layer.norm1(x), tgt_mask)
                cross_output, cross_attention = layer.multihead_attn(
                    layer.norm2(x),
                    memory,
                    memory,
                    need_weights=True,
                    average_attn_weights=True,
                )
                x = x + layer.dropout2(cross_output)
                x = x + self._feed_forward_block(layer, layer.norm3(x))
            else:
                x = layer.norm1(x + self._self_attention_block(layer, x, tgt_mask))
                cross_output, cross_attention = layer.multihead_attn(
                    x,
                    memory,
                    memory,
                    need_weights=True,
                    average_attn_weights=True,
                )
                x = layer.norm2(x + layer.dropout2(cross_output))
                x = layer.norm3(x + self._feed_forward_block(layer, x))

        norm = self.model.transformer.decoder.norm
        if norm is not None:
            x = norm(x)
        if cross_attention is not None:
            cross_attention = cross_attention[0]
        return x, cross_attention

    def _self_attention_block(
        self,
        layer: nn.TransformerDecoderLayer,
        value: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = layer.self_attn(
            value,
            value,
            value,
            attn_mask=tgt_mask,
            need_weights=False,
        )[0]
        return layer.dropout1(output)

    def _feed_forward_block(
        self,
        layer: nn.TransformerDecoderLayer,
        value: torch.Tensor,
    ) -> torch.Tensor:
        value = layer.linear2(layer.dropout(layer.activation(layer.linear1(value))))
        return layer.dropout3(value)

    def _token_label(self, token_id: int) -> str:
        try:
            piece = self.tokenizer.id_to_piece(int(token_id))
        except AttributeError:
            piece = self.tokenizer.IdToPiece(int(token_id))
        return str(piece).replace("▁", " ").strip() or str(piece)

    def _normalize_attention_row(self, row: list[float]) -> list[float]:
        if not row:
            return []
        max_value = max(row)
        if max_value <= 0:
            return [0.0 for _ in row]
        return [round(float(value) / float(max_value), 4) for value in row]


class EncoderBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        dropout: float,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc_h = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc_c = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(src))
        _, (hidden, cell) = self.rnn(embedded)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        cell_cat = torch.cat((cell[-2], cell[-1]), dim=1)
        hidden = torch.tanh(self.fc_h(hidden_cat))
        cell = torch.tanh(self.fc_c(cell_cat))
        return hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        dec_hid_dim: int,
        dropout: float,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.output_dim = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear(dec_hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_token = input_token.unsqueeze(0)
        embedded = self.dropout(self.embedding(input_token))
        output, (hidden, cell) = self.rnn(embedded, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden.squeeze(0), cell.squeeze(0)


class Seq2SeqBiLSTM(nn.Module):
    def __init__(
        self,
        encoder: EncoderBiLSTM,
        decoder: DecoderLSTM,
        device: Any,
        sos_id: int,
        eos_id: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        raise RuntimeError("Training forward pass is not used by the demo API.")

    def translate(self, src: torch.Tensor, max_len: int = 100) -> list[int]:
        self.eval()
        generated_ids = [self.sos_id]

        with torch.inference_mode():
            hidden, cell = self.encoder(src)
            input_token = torch.tensor([self.sos_id], device=self.device)

            for _ in range(max_len):
                output, hidden, cell = self.decoder(input_token, hidden, cell)
                pred_token = output.argmax(1).item()
                if pred_token == self.eos_id:
                    break
                generated_ids.append(pred_token)
                input_token = torch.tensor([pred_token], device=self.device)

        return generated_ids[1:]

    def beam_search(
        self,
        src: torch.Tensor,
        max_len: int = 100,
        beam_size: int = DEFAULT_BEAM_SIZE,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
    ) -> list[int]:
        self.eval()

        with torch.inference_mode():
            hidden, cell = self.encoder(src)
            beams: list[tuple[list[int], float, torch.Tensor, torch.Tensor, bool]] = [
                ([self.sos_id], 0.0, hidden, cell, False)
            ]

            for _ in range(max_len):
                candidates: list[
                    tuple[list[int], float, torch.Tensor, torch.Tensor, bool]
                ] = []

                for token_ids, score, beam_hidden, beam_cell, is_finished in beams:
                    if is_finished:
                        candidates.append(
                            (token_ids, score, beam_hidden, beam_cell, is_finished)
                        )
                        continue

                    input_token = torch.tensor([token_ids[-1]], device=self.device)
                    output, next_hidden, next_cell = self.decoder(
                        input_token,
                        beam_hidden,
                        beam_cell,
                    )
                    log_probs = torch.log_softmax(output[0], dim=-1)
                    top_scores, top_ids = torch.topk(log_probs, beam_size)

                    for token_score, token_id in zip(top_scores.tolist(), top_ids.tolist()):
                        next_id = int(token_id)
                        candidates.append(
                            (
                                [*token_ids, next_id],
                                score + float(token_score),
                                next_hidden,
                                next_cell,
                                next_id == self.eos_id,
                            )
                        )

                beams = sorted(
                    candidates,
                    key=lambda beam: normalize_beam_score(
                        beam[1],
                        len(beam[0]) - 1,
                        length_penalty,
                    ),
                    reverse=True,
                )[:beam_size]
                if all(is_finished for *_, is_finished in beams):
                    break

        best_tokens = max(
            beams,
            key=lambda beam: normalize_beam_score(
                beam[1],
                len(beam[0]) - 1,
                length_penalty,
            ),
        )[0]
        return [token_id for token_id in best_tokens[1:] if token_id != self.eos_id]


class LocalBiLstmTranslator:
    def __init__(
        self,
        weight_path: Path,
        tokenizer_path: Path,
        device_name: str,
    ) -> None:
        import __main__

        self.device = torch.device(device_name)
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.pad_id = self.tokenizer.piece_to_id("<pad>")
        self.sos_id = self.tokenizer.piece_to_id("<sos>")
        self.eos_id = self.tokenizer.piece_to_id("<eos>")

        aliases = {
            "EncoderBiLSTM": EncoderBiLSTM,
            "DecoderLSTM": DecoderLSTM,
            "Seq2SeqBiLSTM": Seq2SeqBiLSTM,
        }
        previous = {name: getattr(__main__, name, None) for name in aliases}
        try:
            for name, value in aliases.items():
                setattr(__main__, name, value)
            self.model = torch.load(weight_path, map_location=self.device, weights_only=False)
        finally:
            for name, value in previous.items():
                if value is None:
                    delattr(__main__, name)
                else:
                    setattr(__main__, name, value)

        self.model.device = self.device
        self.model.sos_id = self.sos_id
        self.model.eos_id = self.eos_id
        self.model.to(self.device)
        self.model.eval()

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
        temperature: float,
        use_beam_search: bool = True,
    ) -> str:
        if source_lang != "en" or target_lang != "vi":
            raise ValueError("This local BiLSTM only supports English to Vietnamese.")

        cleaned = clean_and_tokenize_text(text)
        token_ids = [self.sos_id, *self.tokenizer.encode(cleaned, out_type=int), self.eos_id]
        src = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(1)
        max_len = min(max_new_tokens, LOCAL_MAX_NEW_TOKENS)
        if use_beam_search:
            output_ids = self.model.beam_search(
                src,
                max_len=max_len,
                beam_size=DEFAULT_BEAM_SIZE,
                length_penalty=DEFAULT_LENGTH_PENALTY,
            )
        else:
            output_ids = self.model.translate(src, max_len=max_len)
        return self._decode(output_ids)

    def _decode(self, token_ids: list[int]) -> str:
        special_ids = {self.pad_id, self.sos_id, self.eos_id}
        clean_ids = [token_id for token_id in token_ids if token_id not in special_ids]
        return self.tokenizer.decode(clean_ids).strip()
