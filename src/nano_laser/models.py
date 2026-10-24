# Adapted from laser_encoders.models (LASER, BSD license, Copyright (c) Facebook, Inc. and its affiliates).
# The fairseq TransformerEncoder (0.12.2) and Dictionary are re-implemented below in plain torch, reduced to
# what LASER inference uses. The checkpoints and the cache directory are the ones laser_encoders downloads.

import argparse
import logging
import math
import os
import re
import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nano_laser.download_models import LaserModelDownloader
from nano_laser.language_list import LASER2_LANGUAGE, LASER3_LANGUAGE
from nano_laser.laser_tokenizer import LaserTokenizer, initialize_tokenizer

SPACE_NORMALIZER = re.compile(r"\s+")
Batch = namedtuple("Batch", "srcs tokens lengths")
BOS, PAD, EOS, UNK = 0, 1, 2, 3  # fairseq Dictionary's special symbols

logger = logging.getLogger(__name__)


def load_checkpoint(path) -> dict:
    """The LASER checkpoints only pickle tensors, dicts and an argparse.Namespace: no fairseq class is needed."""
    with torch.serialization.safe_globals([argparse.Namespace]):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_dictionary(path, add_mask: bool = False) -> list[str]:
    """fairseq Dictionary.load(path).symbols: the four special symbols, then one `symbol count` per line.

    A line flagged `#fairseq:overwrite` appends its symbol again, so the symbols can repeat: the embedding
    has one row per symbol, and the index of a repeated symbol is its last one.
    """
    symbols = ["<s>", "<pad>", "</s>", "<unk>"]
    seen = set(symbols)
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, field = line.rstrip().rsplit(" ", 1)
            overwrite = field == "#fairseq:overwrite"
            if overwrite:
                word, field = word.rsplit(" ", 1)
            int(field)  # the count, unused at inference
            if word in seen and not overwrite:
                raise RuntimeError(f"Duplicate word found when loading Dictionary: '{word}'.")
            symbols.append(word)
            seen.add(word)
    if add_mask and "<mask>" not in seen:
        symbols.append("<mask>")
    return symbols


class SentenceEncoder:
    def __init__(
        self,
        model_path,
        max_sentences=None,
        max_tokens=None,
        spm_vocab=None,
        spm_model=None,
        cpu=False,
        fp16=False,
        verbose=False,
        sort_kind="quicksort",
    ):
        if verbose:
            logger.info(f"loading encoder: {model_path}")
        self.spm_model = spm_model
        if self.spm_model:
            self.tokenizer = LaserTokenizer(spm_model=Path(self.spm_model))

        self.use_cuda = torch.cuda.is_available() and not cpu
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens
        if self.max_tokens is None and self.max_sentences is None:
            self.max_sentences = 1

        state_dict = load_checkpoint(model_path)
        if "params" in state_dict:
            self.encoder = LaserLstmEncoder(**state_dict["params"])
            self.encoder.load_state_dict(state_dict["model"])
            self.dictionary = state_dict["dictionary"]
            self.prepend_bos = False
            self.left_padding = False
        else:
            self.encoder = LaserTransformerEncoder(state_dict, spm_vocab)
            self.dictionary = self.encoder.dictionary
            self.prepend_bos = state_dict["cfg"]["model"].prepend_bos
            # stored as the string "True" in the LASER3 checkpoints; truthy, as in laser_encoders
            self.left_padding = state_dict["cfg"]["model"].left_pad_source
        del state_dict
        self.bos_index = self.dictionary["<s>"] = BOS
        self.pad_index = self.dictionary["<pad>"] = PAD
        self.eos_index = self.dictionary["</s>"] = EOS
        self.unk_index = self.dictionary["<unk>"] = UNK

        if fp16:
            self.encoder.half()
        if self.use_cuda:
            if verbose:
                logger.info("transfer encoder to GPU")
            self.encoder.cuda()
        self.encoder.eval()
        self.sort_kind = sort_kind

    def __call__(self, text_or_batch):
        if not self.spm_model:
            raise ValueError(
                "Either initialize the encoder with an spm_model or pre-tokenize and use the encode_sentences method."
            )
        text_or_batch = self.tokenizer(text_or_batch)
        if isinstance(text_or_batch, str):
            text_or_batch = [text_or_batch]
        return self.encode_sentences(text_or_batch)

    def _process_batch(self, batch):
        tokens = batch.tokens
        lengths = batch.lengths
        if self.use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()

        with torch.no_grad():
            sentemb = self.encoder(tokens, lengths)["sentemb"]
        return sentemb.detach().cpu().numpy()

    def _tokenize(self, line):
        tokens = SPACE_NORMALIZER.sub(" ", line).strip().split()
        ids = [self.dictionary.get(token, self.unk_index) for token in tokens] + [self.eos_index]
        if self.prepend_bos:
            ids = [self.bos_index, *ids]
        return torch.LongTensor(ids)

    def _make_batches(self, lines):
        tokens = [self._tokenize(line) for line in lines]
        lengths = np.array([t.numel() for t in tokens])
        indices = np.argsort(-lengths, kind=self.sort_kind)

        def batch(tokens, lengths, indices):
            toks = tokens[0].new_full((len(tokens), tokens[0].shape[0]), self.pad_index)
            for i in range(len(tokens)):
                if self.left_padding:
                    toks[i, -tokens[i].shape[0] :] = tokens[i]
                else:
                    toks[i, : tokens[i].shape[0]] = tokens[i]
            return Batch(srcs=None, tokens=toks, lengths=torch.LongTensor(lengths)), indices

        batch_tokens, batch_lengths, batch_indices = [], [], []
        ntokens = nsentences = 0
        for i in indices:
            if nsentences > 0 and (
                (self.max_tokens is not None and ntokens + lengths[i] > self.max_tokens)
                or (self.max_sentences is not None and nsentences == self.max_sentences)
            ):
                yield batch(batch_tokens, batch_lengths, batch_indices)
                ntokens = nsentences = 0
                batch_tokens, batch_lengths, batch_indices = [], [], []
            batch_tokens.append(tokens[i])
            batch_lengths.append(lengths[i])
            batch_indices.append(i)
            ntokens += tokens[i].shape[0]
            nsentences += 1
        if nsentences > 0:
            yield batch(batch_tokens, batch_lengths, batch_indices)

    def encode_sentences(self, sentences, normalize_embeddings=False):
        indices = []
        results = []
        for batch, batch_indices in self._make_batches(sentences):
            indices.extend(batch_indices)
            encoded_batch = self._process_batch(batch)
            if normalize_embeddings:
                # Perform L2 normalization on the embeddings
                norms = np.linalg.norm(encoded_batch, axis=1, keepdims=True)
                encoded_batch = encoded_batch / norms
            results.append(encoded_batch)
        return np.vstack(results)[np.argsort(indices, kind=self.sort_kind)]


class SinusoidalPositionalEmbedding(nn.Module):
    """fairseq SinusoidalPositionalEmbedding, without incremental decoding."""

    def __init__(self, embedding_dim, padding_idx, init_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = self.get_embedding(init_size, embedding_dim, padding_idx)

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        emb[padding_idx, :] = 0
        return emb

    def forward(self, tokens, like):
        bsz, seq_len = tokens.shape
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):  # computed on cpu, as fairseq does
            self.weights = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(like)
        # fairseq utils.make_positions: padding keeps padding_idx, the tokens count from padding_idx + 1
        mask = tokens.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1)


class MultiheadAttention(nn.Module):
    """fairseq MultiheadAttention for encoder self-attention, without dropout."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, key_padding_mask):  # x: T x B x C, key_padding_mask: B x T, True on padding
        tgt_len, bsz, embed_dim = x.shape

        def split_heads(y):  # T x B x C -> B x H x T x C/H
            return y.view(tgt_len, bsz, self.num_heads, -1).permute(1, 2, 0, 3)

        q, k, v = split_heads(self.q_proj(x)), split_heads(self.k_proj(x)), split_heads(self.v_proj(x))
        attn_mask = None if key_padding_mask is None else ~key_padding_mask[:, None, None, :]
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out_proj(attn.permute(2, 0, 1, 3).reshape(tgt_len, bsz, embed_dim))


class TransformerEncoderLayer(nn.Module):
    """fairseq TransformerEncoderLayerBase at inference."""

    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg.encoder_embed_dim
        self.normalize_before = cfg.encoder_normalize_before
        self.activation_fn = {"relu": F.relu, "gelu": F.gelu}[cfg.activation_fn]
        self.self_attn = MultiheadAttention(embed_dim, cfg.encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, cfg.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(cfg.encoder_ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, encoder_padding_mask):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x = residual + self.self_attn(x, encoder_padding_mask)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = residual + self.fc2(self.activation_fn(self.fc1(x)))
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class LaserTransformerEncoder(nn.Module):
    """fairseq TransformerEncoder (0.12.2) at inference, followed by the LASER sentence pooling."""

    def __init__(self, state_dict, vocab_path):
        super().__init__()
        weights = state_dict["model"]
        symbols = load_dictionary(
            vocab_path, add_mask=any(k in weights for k in ["encoder.layer_norm.weight", "layer_norm.weight"])
        )
        self.dictionary = {symbol: index for index, symbol in enumerate(symbols)}
        if "decoder.version" in weights:  # a full translation model: keep the encoder, without its prefix
            weights = {
                k.removeprefix("encoder."): v
                for k, v in weights.items()
                if k.startswith(("encoder.layer_norm", "encoder.layers", "encoder.embed", "encoder.version"))
            }
        cfg = state_dict["cfg"]["model"]
        unsupported = {
            "encoder_learned_pos": False,
            "no_token_positional_embeddings": False,
            "layernorm_embedding": False,
        }
        for name, value in unsupported.items():
            if getattr(cfg, name, value) != value:
                raise NotImplementedError(f"{name}={getattr(cfg, name)} is not supported")

        embed_dim = cfg.encoder_embed_dim
        self.sentemb_criterion = cfg.sentemb_criterion
        self.padding_idx = PAD
        self.bos_idx = BOS
        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)
        self.embed_tokens = nn.Embedding(len(symbols), embed_dim, padding_idx=self.padding_idx)
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim, self.padding_idx, init_size=cfg.max_source_positions + self.padding_idx + 1
        )
        self.layers = nn.ModuleList(TransformerEncoderLayer(cfg) for _ in range(cfg.encoder_layers))
        self.layer_norm = nn.LayerNorm(embed_dim) if "layer_norm.weight" in weights else None
        # fairseq's version counters and the positional embedding's dtype marker are not weights
        self.load_state_dict(
            {k: v for k, v in weights.items() if not k.endswith(("version", "_float_tensor"))}
        )

    def forward(self, src_tokens, src_lengths):
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = bool(encoder_padding_mask.any())

        x = self.embed_scale * self.embed_tokens(src_tokens)
        x = x + self.embed_positions(src_tokens, like=x)
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        x = x.transpose(0, 1)  # B x T x C -> T x B x C
        for layer in self.layers:
            x = layer(x, encoder_padding_mask if has_pads else None)
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.sentemb_criterion == "cls":
            # the <s> of each sentence; laser_encoders' boolean indexing reorders them once padded
            bos_positions = src_tokens.eq(self.bos_idx).int().argmax(dim=1)
            sentemb = x[bos_positions, torch.arange(x.size(1), device=x.device)]
        else:
            padding_mask = encoder_padding_mask.t().unsqueeze(-1)
            if has_pads:
                x = x.float().masked_fill_(padding_mask, float("-inf")).type_as(x)
            sentemb = x.max(dim=0)[0]
        return {"sentemb": sentemb}


class LaserLstmEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings,
        padding_idx,
        embed_dim=320,
        hidden_size=512,
        num_layers=1,
        bidirectional=False,
        left_pad=True,
        padding_value=0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(num_embeddings, embed_dim, padding_idx=self.padding_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                return torch.cat(
                    [
                        torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(1, bsz, self.output_units)
                        for i in range(self.num_layers)
                    ],
                    dim=0,
                )

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = encoder_padding_mask.unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float("-inf")).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        return {
            "sentemb": sentemb,
            "encoder_out": (x, final_hiddens, final_cells),
            "encoder_padding_mask": encoder_padding_mask if encoder_padding_mask.any() else None,
        }


def initialize_encoder(
    lang: str | None = None,
    model_dir: str | None = None,
    spm: bool = True,
    laser: str | None = None,
    **encoder_kwargs,
):
    """`encoder_kwargs` go to SentenceEncoder (max_sentences, max_tokens, cpu, fp16, ...)."""
    downloader = LaserModelDownloader(model_dir)
    if laser is not None:
        if laser == "laser3":
            lang = downloader.get_language_code(LASER3_LANGUAGE, lang)
            downloader.download_laser3(lang=lang, spm=spm)
            file_path = f"laser3-{lang}.v1"
        elif laser == "laser2":
            downloader.download_laser2()
            file_path = "laser2"
        else:
            raise ValueError(f"Unsupported laser model: {laser}. Choose either laser2 or laser3.")
    elif lang in LASER3_LANGUAGE:
        lang = downloader.get_language_code(LASER3_LANGUAGE, lang)
        downloader.download_laser3(lang=lang, spm=spm)
        file_path = f"laser3-{lang}.v1"
    elif lang in LASER2_LANGUAGE:
        downloader.download_laser2()
        file_path = "laser2"
    else:
        raise ValueError(f"Unsupported language name: {lang}. Please specify a supported language name.")

    model_dir = downloader.model_dir
    model_path = os.path.join(model_dir, f"{file_path}.pt")
    spm_vocab = os.path.join(model_dir, f"{file_path}.cvocab")

    if not os.path.exists(spm_vocab):
        # if there is no cvocab for the laser3 lang use laser2 cvocab
        spm_vocab = os.path.join(model_dir, "laser2.cvocab")

    return SentenceEncoder(model_path=model_path, spm_vocab=spm_vocab, spm_model=None, **encoder_kwargs)


class LaserEncoderPipeline:
    def __init__(
        self,
        lang: str | None = None,
        model_dir: str | None = None,
        spm: bool = True,
        laser: str | None = None,
        **encoder_kwargs,
    ):
        if laser == "laser2" and lang is not None:
            warnings.warn(
                "Warning: The 'lang' parameter is optional when using 'laser2'. It will be ignored.",
                stacklevel=2,
            )

        if laser == "laser3" and lang is None:
            raise ValueError("For 'laser3', the 'lang' parameter is required.")

        if laser is None and lang is None:
            raise ValueError("Either 'laser' or 'lang' should be provided.")

        self.tokenizer = initialize_tokenizer(lang=lang, model_dir=model_dir, laser=laser)
        self.encoder = initialize_encoder(
            lang=lang, model_dir=model_dir, spm=spm, laser=laser, **encoder_kwargs
        )

    def encode_sentences(self, sentences: list, normalize_embeddings: bool = False) -> np.ndarray:
        """Tokenizes and encodes a list of sentences, one embedding row per sentence."""
        tokenized_sentences = [self.tokenizer.tokenize(sentence) for sentence in sentences]
        return self.encoder.encode_sentences(tokenized_sentences, normalize_embeddings)
