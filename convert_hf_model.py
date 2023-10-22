import argparse
import os
import struct
import torch
from transformers import AutoModelForCausalLM
from sentencepiece import SentencePieceProcessor


def serialize_f32(file, tensor):
    tensor_f32 = tensor.detach().cpu().view(-1).to(torch.float32).numpy()

    file.write(struct.pack(f"{len(tensor_f32)}f", *tensor_f32))


# https://github.com/huggingface/transformers/blob/5c081e29930466ecf9a478727039d980131076d9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122C28-L122C35
def unpermute(tensor, n_heads, dim1, dim2):
    return (
        tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


def write_checkpoint_file():
    model = AutoModelForCausalLM.from_pretrained(args.input_model_path)

    if model.config.model_type != "llama":
        parser.error("Expected llama model")

    if model.config.rope_theta != 10000:
        parser.error("Expected a RoPE frequency base of 10000")

    state = model.state_dict()
    embedding_weights = state["model.embed_tokens.weight"]
    output_norm_weight = state["model.norm.weight"]
    output_weight = state[f"lm_head.weight"]

    embedding_size = model.config.hidden_size
    ffn_hidden_size = model.config.intermediate_size
    n_layers = model.config.num_hidden_layers
    n_attention_heads = model.config.num_attention_heads
    n_attention_query_groups = model.config.num_key_value_heads
    vocab_size = model.config.vocab_size
    max_sequence_length = model.config.max_position_embeddings
    shared_output_weight = torch.equal(embedding_weights, output_weight)

    os.makedirs(args.output_model_path, exist_ok=True)

    output_file = open(os.path.join(args.output_model_path, "checkpoint_v1.bin"), "wb")

    output_file.write(struct.pack("I", 0x616B3432))
    output_file.write(struct.pack("i", 1))

    output_file.write(
        struct.pack(
            "iiiiiii",
            embedding_size,
            ffn_hidden_size,
            n_layers,
            n_attention_heads,
            n_attention_query_groups,
            vocab_size,
            max_sequence_length,
        )
    )

    output_file.write(struct.pack("B", int(shared_output_weight)))
    output_file.write(b"\0" * (256 - output_file.tell()))

    for layer in range(n_layers):
        attention_norm_weight = state[f"model.layers.{layer}.input_layernorm.weight"]

        serialize_f32(output_file, attention_norm_weight)

    for layer in range(n_layers):
        ffn_norm_weight = state[f"model.layers.{layer}.post_attention_layernorm.weight"]

        serialize_f32(output_file, ffn_norm_weight)

    serialize_f32(output_file, output_norm_weight)
    serialize_f32(output_file, embedding_weights)

    for layer in range(n_layers):
        attention_query_weight = state[f"model.layers.{layer}.self_attn.q_proj.weight"]

        serialize_f32(
            output_file,
            unpermute(
                attention_query_weight,
                n_attention_heads,
                embedding_size,
                embedding_size,
            ),
        )

    for layer in range(n_layers):
        attention_key_weight = state[f"model.layers.{layer}.self_attn.k_proj.weight"]

        if n_attention_heads == n_attention_query_groups:
            serialize_f32(
                output_file,
                unpermute(
                    attention_key_weight,
                    n_attention_heads,
                    embedding_size,
                    embedding_size,
                ),
            )
        else:
            serialize_f32(
                output_file,
                unpermute(
                    attention_key_weight,
                    n_attention_query_groups,
                    embedding_size // n_attention_heads * n_attention_query_groups,
                    embedding_size,
                ),
            )

    for layer in range(n_layers):
        attention_value_weight = state[f"model.layers.{layer}.self_attn.v_proj.weight"]

        serialize_f32(output_file, attention_value_weight)

    for layer in range(n_layers):
        attention_output_weight = state[f"model.layers.{layer}.self_attn.o_proj.weight"]

        serialize_f32(output_file, attention_output_weight)

    for layer in range(n_layers):
        ffn_gate_weight = state[f"model.layers.{layer}.mlp.gate_proj.weight"]

        serialize_f32(output_file, ffn_gate_weight)

    for layer in range(n_layers):
        ffn_down_weight = state[f"model.layers.{layer}.mlp.down_proj.weight"]

        serialize_f32(output_file, ffn_down_weight)

    for layer in range(n_layers):
        ffn_up_weight = state[f"model.layers.{layer}.mlp.up_proj.weight"]

        serialize_f32(output_file, ffn_up_weight)

    if not shared_output_weight:
        serialize_f32(output_file, output_weight)

    output_file.close()


def write_tokenizer_file():
    sp_model = SentencePieceProcessor(
        model_file=os.path.join(args.input_model_path, "tokenizer.model")
    )

    words, scores = [], []

    for token in range(sp_model.vocab_size()):
        word = sp_model.id_to_piece(token)
        score = sp_model.get_score(token)

        if token == sp_model.bos_id():
            word = "\n<s>\n"
        elif token == sp_model.eos_id():
            word = "\n</s>\n"

        words.append(word.replace("▁", " ").encode("utf-8"))
        scores.append(score)

    max_word_length = max(len(word) for word in words)

    os.makedirs(args.output_model_path, exist_ok=True)

    output_file = open(os.path.join(args.output_model_path, "tokenizer.bin"), "wb")

    output_file.write(struct.pack("I", max_word_length))

    for word, score in zip(words, scores):
        output_file.write(struct.pack("fI", score, len(word)))
        output_file.write(word)

    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_model_path", type=str, help="the input model")
    parser.add_argument("output_model_path", type=str, help="the output model")

    args = parser.parse_args()

    write_checkpoint_file()
    write_tokenizer_file()
