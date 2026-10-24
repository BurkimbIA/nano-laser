"""Tiny random checkpoints in the LASER formats: no download, runs on cpu."""

import argparse

import numpy as np
import pytest
import torch

from nano_laser.models import SentenceEncoder, load_dictionary

WORDS = [f"▁w{i}" for i in range(40)]


@pytest.fixture
def vocab(tmp_path):
    path = tmp_path / "vocab.cvocab"
    lines = ["<unk> 1 #fairseq:overwrite", "<s> 1 #fairseq:overwrite", "</s> 1 #fairseq:overwrite"]
    path.write_text("\n".join(lines + [f"{w} 1" for w in WORDS]) + "\n", encoding="utf-8")
    return path


def sentences(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS, size=rng.integers(1, 20))) for _ in range(n)]


def transformer_checkpoint(path, vocab, *, normalize_before=False, criterion="maxpool", prepend_bos=False):
    cfg = argparse.Namespace(
        encoder_embed_dim=16,
        encoder_ffn_embed_dim=32,
        encoder_attention_heads=4,
        encoder_layers=2,
        encoder_normalize_before=normalize_before,
        encoder_learned_pos=False,
        no_token_positional_embeddings=False,
        no_scale_embedding=False,
        activation_fn="relu",
        max_source_positions=8,  # shorter than some sentences: the positions table must grow
        sentemb_criterion=criterion,
        prepend_bos=prepend_bos,
        left_pad_source="True",
    )
    torch.manual_seed(0)
    num_symbols = len(load_dictionary(vocab))
    model = {"version": torch.tensor([3.0]), "embed_tokens.weight": torch.randn(num_symbols, 16)}
    model["embed_positions._float_tensor"] = torch.tensor([1.0])
    for i in range(2):
        for name, shape in {"self_attn.q_proj": (16, 16), "self_attn.k_proj": (16, 16), "self_attn.v_proj": (16, 16),
                            "self_attn.out_proj": (16, 16), "fc1": (32, 16), "fc2": (16, 32)}.items():  # fmt: skip
            model[f"layers.{i}.{name}.weight"] = torch.randn(shape) / 4
            model[f"layers.{i}.{name}.bias"] = torch.randn(shape[0]) / 4
        for name in ["self_attn_layer_norm", "final_layer_norm"]:
            model[f"layers.{i}.{name}.weight"] = 1 + torch.randn(16) / 10
            model[f"layers.{i}.{name}.bias"] = torch.randn(16) / 10
    if normalize_before:
        model["layer_norm.weight"], model["layer_norm.bias"] = torch.ones(16), torch.zeros(16)
        model["embed_tokens.weight"] = torch.randn(num_symbols + 1, 16)  # with <mask>
    torch.save({"cfg": {"model": cfg}, "model": model}, path)
    return path


def lstm_checkpoint(path):
    torch.manual_seed(0)
    params = dict(num_embeddings=4 + len(WORDS), padding_idx=1, embed_dim=8, hidden_size=6, num_layers=2,
                  bidirectional=True, left_pad=True, padding_value=0.0)  # fmt: skip
    lstm = torch.nn.LSTM(8, 6, num_layers=2, bidirectional=True)
    model = {"embed_tokens.weight": torch.randn(params["num_embeddings"], 8)}
    model.update({f"lstm.{k}": v for k, v in lstm.state_dict().items()})
    dictionary = {w: i + 4 for i, w in enumerate(WORDS)}
    torch.save({"params": params, "model": model, "dictionary": dictionary}, path)
    return path


def test_load_dictionary_keeps_overwritten_symbols(vocab):
    symbols = load_dictionary(vocab)
    assert symbols[:7] == ["<s>", "<pad>", "</s>", "<unk>", "<unk>", "<s>", "</s>"]
    assert len(symbols) == 7 + len(WORDS)
    assert load_dictionary(vocab, add_mask=True)[-1] == "<mask>"


def test_load_dictionary_rejects_duplicates(tmp_path):
    path = tmp_path / "dup.cvocab"
    path.write_text("▁a 1\n▁a 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Duplicate"):
        load_dictionary(path)


@pytest.mark.parametrize(
    "options",
    [{}, {"normalize_before": True}, {"criterion": "cls", "prepend_bos": True}],
    ids=["post-norm", "pre-norm", "cls"],
)
def test_transformer_batching_does_not_change_embeddings(tmp_path, vocab, options):
    path = transformer_checkpoint(tmp_path / "laser3.pt", vocab, **options)
    one = SentenceEncoder(path, spm_vocab=vocab, cpu=True).encode_sentences(sentences())
    batched = SentenceEncoder(path, spm_vocab=vocab, cpu=True, max_sentences=5).encode_sentences(sentences())
    assert one.shape == (12, 16)
    np.testing.assert_allclose(batched, one, rtol=1e-5, atol=1e-5)


def test_transformer_left_pads(tmp_path, vocab):
    encoder = SentenceEncoder(
        transformer_checkpoint(tmp_path / "laser3.pt", vocab), spm_vocab=vocab, cpu=True, max_sentences=2
    )
    batch, _ = next(encoder._make_batches(["▁w1", "▁w1 ▁w2 ▁w3"]))
    assert batch.tokens.tolist() == [[8, 9, 10, 2], [1, 1, 8, 2]]


def test_lstm_batching_does_not_change_embeddings(tmp_path):
    path = lstm_checkpoint(tmp_path / "laser2.pt")
    one = SentenceEncoder(path, cpu=True).encode_sentences(sentences())
    batched = SentenceEncoder(path, cpu=True, max_tokens=40).encode_sentences(sentences())
    assert one.shape == (12, 12)
    np.testing.assert_allclose(batched, one, rtol=1e-6, atol=1e-6)


def test_normalize_embeddings(tmp_path):
    encoder = SentenceEncoder(lstm_checkpoint(tmp_path / "laser2.pt"), cpu=True)
    embeddings = encoder.encode_sentences(sentences(), normalize_embeddings=True)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, rtol=1e-6)


def test_unknown_words_map_to_unk(tmp_path):
    encoder = SentenceEncoder(lstm_checkpoint(tmp_path / "laser2.pt"), cpu=True)
    assert encoder._tokenize("▁w0  ▁nope").tolist() == [4, 3, 2]
