"""
Script to encode natural language instructions into text embeddings using SigLIP model.
This script processes instruction datasets and generates embeddings for robot manipulation tasks.
"""

import argparse
import json
import pathlib
import torch
import yaml
from helpers.common import ACDIT_DEFAULT_LOGGER
from models.multimodal_encoder.siglip_lang_encoder import SiglipLangEncoder


def main(args):
    """
    Main function to process instruction datasets and generate text embeddings.
    
    Args:
        args: Command line arguments containing config path, dataset root, GPU ID, and overwrite flag.
    """
    # Set up device for computation
    device = torch.device(f"cuda:{args.gpu_id}")
    dataset_root = pathlib.Path(args.dataset_root)
    
    # Load configuration file containing tasks and instructions
    with open(args.config, "r") as f:
        config = json.load(f)

    # Initialize SigLIP language encoder for text embedding (once for all tasks)
    cfg_path = pathlib.Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(cfg_path, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    text_embedder = SiglipLangEncoder(
        from_pretrained="google/siglip-so400m-patch14-384",
        model_max_length=cfg["dataset"]["tokenizer_max_length"],
        device=device,
        use_offload_folder=None,
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    # Process each task configuration
    for item in config:
        task = item["task"]
        subtask = item["subtask"]
        object = item["object"]
        instructions = item["instructions"]
        
        # Create directory structure for the current task
        dataset_dir = dataset_root / task / subtask / "train" / object
        dataset_instructions_dir = dataset_dir / "instructions"

        # Check if instructions directory already exists
        if dataset_instructions_dir.exists():
            if args.overwrite:
                ACDIT_DEFAULT_LOGGER.warning(f"Instructions for {task}/{subtask}/{object} already exist. Overwriting.")
            else:
                ACDIT_DEFAULT_LOGGER.warning(f"Instructions for {task}/{subtask}/{object} already exist. Skipping.")
                continue
        else:
            # Create the instructions directory
            dataset_instructions_dir.mkdir(parents=True, exist_ok=True)
            ACDIT_DEFAULT_LOGGER.info(f"Creating instructions directory for {task}/{subtask}/{object}.")

        # Tokenize all instructions for the current task
        tokenized_res = tokenizer(instructions, return_tensors="pt", padding="longest", truncation=True)
        tokens = tokenized_res["input_ids"].to(device)

        # Generate text embeddings using the encoder
        with torch.no_grad():
            text_embeds = text_encoder(input_ids=tokens)["last_hidden_state"].detach().cpu()

        # Save individual embeddings for each instruction
        for i in range(len(instructions)):
            text_embed = text_embeds[i]
            save_path = dataset_instructions_dir / f"lang_embed_{i}.pt"
            
            # Create item dictionary with metadata and embedding
            item = {
                "task_name": "_".join((task, subtask, object)),
                "instruction": instructions[i],
                "text_embed": text_embed,
            }
            # Save the embedding to file
            torch.save(item, save_path)
            ACDIT_DEFAULT_LOGGER.info(f"Saved embedding for {task}/{subtask}/{object} instruction {i} to {save_path}")

    ACDIT_DEFAULT_LOGGER.ok("Done!")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Encode natural language instructions into text embeddings")
    parser.add_argument("--config", type=str, 
                       default=str(pathlib.Path(__file__).parents[2] / "configs" / "mshab_languages.json"),
                       help="Path to configuration file containing tasks and instructions")
    parser.add_argument("--dataset-root", type=str, required=True,
                       help="Root directory where the dataset will be stored")
    parser.add_argument("--gpu-id", type=int, default=0,
                       help="GPU device ID to use for computation")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing instruction embeddings if they exist")
    main(parser.parse_args())

# Example usage:
# python -m data.mshab.encode_instructions --dataset-root /share/project/chensixiang/Codes/AC-DiT/third_party/mshab/mshab_data/gen_data_save_trajectories --overwrite
