import argparse
import os
import struct
import torch
from transformers import AutoModelForCausalLM
from sentencepiece import SentencePieceProcessor


def serialize_f32(file, tensor):
    tensor_f32 = tensor.detach().cpu().view(-1).to(torch.float32).numpy()

    file.write(struct.pack(f"{len(tensor_f32)}f", *tensor_f32))


def write_checkpoint_file():
    hf_model = AutoModelForCausalLM.from_pretrained(args.input_model_path)

    if hf_model.config.model_type != "llama":
        parser.error("Expected llama model")

    if hf_model.config.rope_theta != 10000:
        parser.error("Expected a RoPE frequency base of 10000")

    hf_state_dict = hf_model.state_dict()
    token_embedding_vectors = hf_state_dict["model.embed_tokens.weight"]
    output_matrix = hf_state_dict[f"lm_head.weight"]

    embedding_size = hf_model.config.hidden_size
    ffn_hidden_size = hf_model.config.intermediate_size
    n_layers = hf_model.config.num_hidden_layers
    n_attention_heads = hf_model.config.num_attention_heads
    n_attention_query_groups = hf_model.config.num_key_value_heads
    vocab_size = hf_model.config.vocab_size
    max_sequence_length = hf_model.config.max_position_embeddings
    shared_output_matrix = torch.equal(token_embedding_vectors, output_matrix)

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

    output_file.write(struct.pack("B", int(shared_output_matrix)))
    output_file.write(b"\0" * (256 - output_file.tell()))

    # attention_norm_vectors
    for layer in range(n_layers):
        serialize_f32(
            output_file, hf_state_dict[f"model.layers.{layer}.input_layernorm.weight"]
        )

    # ffn_norm_vectors
    for layer in range(n_layers):
        serialize_f32(
            output_file,
            hf_state_dict[f"model.layers.{layer}.post_attention_layernorm.weight"],
        )

    # output_norm_vector
    serialize_f32(output_file, hf_state_dict["model.norm.weight"])

    serialize_f32(output_file, token_embedding_vectors)

    # https://github.com/huggingface/transformers/blob/5c081e29930466ecf9a478727039d980131076d9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122C28-L122C35
    def unpermute(tensor):
        return (
            tensor.view(
                n_attention_heads,
                2,
                embedding_size // n_attention_heads // 2,
                embedding_size,
            )
            .transpose(1, 2)
            .reshape(embedding_size, embedding_size)
        )

    # attention_query_matrices
    for layer in range(n_layers):
        serialize_f32(
            output_file,
            unpermute(hf_state_dict[f"model.layers.{layer}.self_attn.q_proj.weight"]),
        )

    # attention_key_matrices
    for layer in range(n_layers):
        serialize_f32(
            output_file,
            unpermute(hf_state_dict[f"model.layers.{layer}.self_attn.k_proj.weight"]),
        )

    # attention_value_matrices
    for layer in range(n_layers):
        serialize_f32(
            output_file, hf_state_dict[f"model.layers.{layer}.self_attn.v_proj.weight"]
        )

    # attention_output_matrices
    for layer in range(n_layers):
        serialize_f32(
            output_file, hf_state_dict[f"model.layers.{layer}.self_attn.o_proj.weight"]
        )

    # ffn_gate_matrices
    for layer in range(n_layers):
        serialize_f32(
            output_file, hf_state_dict[f"model.layers.{layer}.mlp.gate_proj.weight"]
        )

    # ffn_down_matrices
    for layer in range(n_layers):
        serialize_f32(
            output_file, hf_state_dict[f"model.layers.{layer}.mlp.down_proj.weight"]
        )

    # ffn_up_matrices
    for layer in range(n_layers):
        serialize_f32(
            output_file, hf_state_dict[f"model.layers.{layer}.mlp.up_proj.weight"]
        )

    if not shared_output_matrix:
        serialize_f32(output_file, output_matrix)

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
