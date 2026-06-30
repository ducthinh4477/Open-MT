import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class EncoderBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, dropout, pad_id):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)

        self.rnn = nn.LSTM(
            emb_dim,
            enc_hid_dim,
            bidirectional=True
        )

        self.fc_h = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc_c = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [src_len, batch]

        embedded = self.dropout(self.embedding(src))
        # embedded: [src_len, batch, emb_dim]

        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [src_len, batch, enc_hid_dim * 2]
        # hidden: [2, batch, enc_hid_dim]

        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        cell_cat = torch.cat((cell[-2], cell[-1]), dim=1)

        hidden = torch.tanh(self.fc_h(hidden_cat))
        cell = torch.tanh(self.fc_c(cell_cat))

        # hidden/cell: [batch, dec_hid_dim]
        return hidden, cell
class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, dec_hid_dim, dropout, pad_id):
        super().__init__()

        self.output_dim = vocab_size

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)

        self.rnn = nn.LSTM(
            emb_dim,
            dec_hid_dim
        )

        self.fc_out = nn.Linear(dec_hid_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell):
        # input_token: [batch]
        input_token = input_token.unsqueeze(0)
        # input_token: [1, batch]

        embedded = self.dropout(self.embedding(input_token))
        # embedded: [1, batch, emb_dim]

        output, (hidden, cell) = self.rnn(
            embedded,
            (hidden.unsqueeze(0), cell.unsqueeze(0))
        )

        prediction = self.fc_out(output.squeeze(0))
        # prediction: [batch, vocab_size]

        return prediction, hidden.squeeze(0), cell.squeeze(0)
class Seq2SeqBiLSTM(nn.Module):
    def __init__(self, encoder, decoder, device, sos_id, eos_id):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len, batch_size = trg.shape
        vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        input_token = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)

            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input_token = trg[t] if teacher_force else top1

        return outputs

    def translate(self, src, max_len=100):
        # src: [src_len, 1]
        self.eval()

        generated_ids = [self.sos_id]

        with torch.no_grad():
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