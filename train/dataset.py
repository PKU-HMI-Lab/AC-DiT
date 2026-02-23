"""
VLA Consumer Dataset Module

This module implements a PyTorch dataset for Vision-Language-Action (VLA) models,
specifically designed for robot manipulation tasks. It handles multi-modal data including:
- Visual observations from multiple cameras
- Robot state and action sequences  
- Natural language instructions
- Point cloud data
- In-context conditions for goal-oriented manipulation

The dataset supports both HDF5-based data loading and buffer-based data loading,
with comprehensive data augmentation and preprocessing capabilities.
"""

import traceback
import json
import math
import random
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import transformers

from data.hdf5_mshab_dataset import HDF5MSHABDataset
from train.image_corrupt import image_corrupt


class VLAConsumerDataset(Dataset):
    """
    A PyTorch dataset for Vision-Language-Action (VLA) model training.
    
    This dataset handles multi-modal robotic data including visual observations,
    robot states/actions, natural language instructions, and contextual information.
    It supports various data augmentation techniques and can work with both
    HDF5-based and buffer-based data storage.
    
    Args:
        config: Dataset configuration dictionary
        tokenizer: Text tokenizer for processing language instructions
        image_processor: Image processor for visual observations
        num_cameras: Number of camera viewpoints
        img_history_size: Number of historical images to include
        image_size: Target image size for resizing (optional)
        auto_adjust_image_brightness: Whether to automatically adjust image brightness
        image_aug: Whether to apply image augmentation
        dataset_type: Type of dataset ('pretrain' or 'finetune')
        cond_mask_prob: Probability of masking conditional inputs
        cam_ext_mask_prob: Probability of masking external camera inputs
        state_noise_snr: Signal-to-noise ratio for state noise injection
        use_hdf5: Whether to use HDF5-based data loading
        hdf5_dataset_name: Name of the HDF5 dataset to use
        use_precomp_lang_embed: Whether to use precomputed language embeddings
        task_names: List of task names to include
    """
    
    def __init__(
        self,
        config,
        tokenizer,
        image_processor,
        num_cameras,
        img_history_size,
        image_size=None,
        auto_adjust_image_brightness=False,
        image_aug=False,
        dataset_type='pretrain',
        cond_mask_prob=0.1,
        cam_ext_mask_prob=-1.0,
        state_noise_snr=None,
        use_hdf5=False,
        hdf5_dataset_name="",
        use_precomp_lang_embed=False,
        base_only=False,
        data_dir=None,
    ):
        """Initialize the VLA Consumer Dataset with specified configuration."""
        super(VLAConsumerDataset, self).__init__()

        # Store config for later use
        self.config = config

        # Extract dataset config
        dataset_config = config["dataset"]

        # Load control frequency mapping for different datasets
        with open("configs/dataset_control_freq.json", 'r') as fp:
            self.control_freq = json.load(fp)

        # Load dataset names based on training type (pretrain vs finetune)
        dataset_names_cfg = 'configs/pretrain_datasets.json' \
            if dataset_type == 'pretrain' else 'configs/finetune_datasets.json'
        with open(dataset_names_cfg, 'r') as file:
            DATASET_NAMES = json.load(file)

        # Create bidirectional mapping between dataset names and IDs
        self.dataset_name2id = {name: i for i, name in enumerate(DATASET_NAMES)}
        self.dataset_id2name = {i: name for i, name in enumerate(DATASET_NAMES)}

        # Store image processor for visual data preprocessing
        self.image_processor = image_processor

        # Dataset configuration parameters
        self.buffer_dir = dataset_config["buf_path"]  # Path to data buffer
        self.num_chunks = dataset_config["buf_num_chunks"]  # Number of data chunks
        self.chunk_size = dataset_config["buf_chunk_size"]  # Size of each chunk
        self.tokenizer_max_length = dataset_config["tokenizer_max_length"]  # Max token length
        self.image_aspect_ratio = dataset_config["image_aspect_ratio"]  # Image aspect ratio handling
        
        # Training and augmentation parameters
        self.state_noise_snr = state_noise_snr  # Noise level for state augmentation
        self.num_cameras = num_cameras  # Number of camera viewpoints
        self.img_history_size = img_history_size  # Historical image context size
        self.cond_mask_prob = cond_mask_prob  # Probability of masking conditions
        self.cam_ext_mask_prob = cam_ext_mask_prob  # External camera masking probability
        
        # Data source configuration
        self.use_hdf5 = use_hdf5
        self.hdf5_dataset = None

        # Initialize HDF5 dataset if specified
        if hdf5_dataset_name == 'mshab':
            self.hdf5_dataset = HDF5MSHABDataset(
                data_dir=data_dir,
                config=config,
                mode="full" if not base_only else "base_only",
            )
        else:
            raise ValueError(f"Unknown hdf5 dataset name {hdf5_dataset_name}")
                
        # Language embedding configuration
        self.use_precomp_lang_embed = use_precomp_lang_embed
        if use_precomp_lang_embed:
            # Load empty language embedding for masking
            self.empty_lang_embed = torch.load("data/empty_lang_embed.pt")
        
        # Load dataset stat
        with open("configs/dataset_stat.json", 'r') as f:
            dataset_stat = json.load(f)
        self.dataset_stat = dataset_stat
        
        # Store additional configuration
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.auto_adjust_image_brightness = auto_adjust_image_brightness
        self.image_aug = image_aug
    
    def get_dataset_name2id(self):
        """Return the mapping from dataset names to IDs."""
        return self.dataset_name2id
    
    def get_dataset_id2name(self):
        """Return the mapping from dataset IDs to names."""
        return self.dataset_id2name
        
    @staticmethod
    def pairwise(iterable):
        """Create pairwise iterator from an iterable (e.g., [a,b,c,d] -> [(a,b), (c,d)])."""
        a = iter(iterable)
        return zip(a, a)

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        if self.use_hdf5:
            return len(self.hdf5_dataset)
        else:
            return self.num_chunks * self.chunk_size
    
    def __getitem__(self, index):
        """
        Get a training sample from the dataset.
        
        This method implements robust data loading with retry logic to handle
        potential data corruption or loading errors. It processes multi-modal
        data including images, robot states, actions, and language instructions.
        
        Args:
            index: Sample index (may be ignored if using HDF5 random sampling)
            
        Returns:
            dict: Training sample containing processed multi-modal data
        """
        # Implement robust data loading with retry logic
        while True:
            data_dict = None
            try:
                # Load raw data from HDF5 dataset
                res = self.hdf5_dataset.get_item()
                content = res['meta']
                states = res['state']
                actions = res['actions']
                state_elem_mask = res['state_indicator']
                
                # Extract visual data with masks indicating valid frames
                image_metas = [
                    res['cam_high'], res['cam_high_mask'],
                    res['cam_right_wrist'], res['cam_right_wrist_mask'],
                    res['cam_left_wrist'], res['cam_left_wrist_mask'],
                ]
                
                # Extract statistical information for normalization
                state_std = res['state_std']
                state_mean = res['state_mean']
                state_norm = res['state_norm']
                pointclouds = res['pointclouds']  # Load point cloud data
                
                # Extract in-context conditions dynamically
                in_context_conditions = res.get('in-context_conditions', {})
            
                # Initialize data dictionary with metadata
                data_dict = {}
                data_dict['dataset_name'] = content['dataset_name']
                data_dict['data_idx'] = self.dataset_name2id[data_dict['dataset_name']]
                
                # Apply conditional masking to control frequency
                data_dict['ctrl_freq'] = self.control_freq[data_dict['dataset_name']] \
                    if random.random() > self.cond_mask_prob else 0
                
                # Apply state noise augmentation if specified
                if self.state_noise_snr is not None:
                    states += np.random.normal(
                        0.0, state_std / np.sqrt(10 ** (self.state_noise_snr / 10)), 
                        states.shape)
                        
                # Prepare state masking with dataset-specific mean states
                ds_state_mean = np.array(self.dataset_stat[data_dict['dataset_name']]['state_mean'])
                ds_state_mean = np.tile(ds_state_mean[None], (states.shape[0], 1))
                
                # Apply conditional masking to states and state indicators
                data_dict["states"] = states \
                    if random.random() > self.cond_mask_prob else ds_state_mean
                data_dict["actions"] = actions
                data_dict["state_elem_mask"] = state_elem_mask \
                    if random.random() > self.cond_mask_prob else np.zeros_like(state_elem_mask)
                
                # Store episode-level statistics
                data_dict["state_norm"] = state_norm

                # Process and concatenate in-context conditions into a single tensor
                # This unifies all contextual information (goal states, task parameters, etc.)
                # into a single tensor for efficient processing by the model
                in_context_values = []
                for condition_key, condition_value in in_context_conditions.items():
                    # Flatten each condition and add to the list
                    flattened_value = condition_value.flatten()
                    in_context_values.append(flattened_value)
                
                # Concatenate all in-context conditions or create empty tensor if none exist
                if in_context_values:
                    data_dict["in_context_conditions"] = np.concatenate(in_context_values, axis=0)
                else:
                    data_dict["in_context_conditions"] = np.array([], dtype=np.float32)
                
                # Prepare background image for invalid/masked frames
                # This creates a consistent background when frames are masked or invalid
                background_color = np.array([
                    int(x*255) for x in self.image_processor.image_mean
                ], dtype=np.uint8).reshape(1, 1, 3)
                background_image = np.ones((
                    self.image_processor.size["height"], 
                    self.image_processor.size["width"], 3), dtype=np.uint8
                ) * background_color
                
                # Process image sequences with conditional masking for data augmentation
                image_metas = list(self.pairwise(image_metas))
                mask_probs = [self.cond_mask_prob] * self.num_cameras
                if self.cam_ext_mask_prob >= 0.0:
                    mask_probs[0] = self.cam_ext_mask_prob  # Special masking for external camera
                    
                # Rearrange images with history and validity masks
                # Format: [history_step_0_cam_0, history_step_0_cam_1, ..., history_step_N_cam_M]
                rearranged_images = []
                for i in range(self.img_history_size):
                    for j in range(self.num_cameras):
                        images, image_mask = image_metas[j]
                        image, valid = images[i], image_mask[i]
                        # Use real image if valid and not masked, otherwise use background
                        if valid and (math.prod(image.shape) > 0) and \
                            (random.random() > mask_probs[j]):
                            rearranged_images.append((image, True))
                        else:
                            rearranged_images.append((background_image.copy(), False))
                
                # Preprocess images with augmentation and formatting
                preprocessed_images = []
                processor = self.image_processor
                for image, valid in rearranged_images:
                    image = Image.fromarray(image)
                    
                    # Resize image if target size is specified
                    if self.image_size is not None:
                        image = transforms.Resize(self.image_size)(image)
                    
                    # Automatic brightness adjustment for dark images
                    if valid and self.auto_adjust_image_brightness:
                        pixel_values = list(image.getdata())
                        average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                        if average_brightness <= 0.15:
                            image = transforms.ColorJitter(brightness=(1.75,1.75))(image)
                    
                    # Apply image augmentation to 50% of valid images
                    if valid and self.image_aug and (random.random() > 0.5):
                        aug_type = random.choice([
                            "corrput_only", "color_only", "both"])
                        if aug_type != "corrput_only":
                            image = transforms.ColorJitter(
                                brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03)(image)
                        if aug_type != "color_only":
                            image = image_corrupt(image)
                    
                    # Handle different aspect ratio policies
                    if self.image_aspect_ratio == 'pad':
                        def expand2square(pil_img, background_color):
                            """Expand image to square by padding with background color."""
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                        
                    # Apply final image preprocessing
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    preprocessed_images.append(image)
                    
                data_dict["images"] = preprocessed_images

                # Store point cloud data
                data_dict["pointclouds"] = pointclouds

                # Process language instructions
                if self.use_precomp_lang_embed:
                    # Use precomputed language embeddings
                    if content["instruction"][-1] == ".":
                        content["instruction"] = content["instruction"][:-1]
                    data_dict["lang_embed"] = torch.load(content["instruction"])["text_embed"] \
                        if random.random() > self.cond_mask_prob else self.empty_lang_embed
                else:
                    # Tokenize instructions on-the-fly
                    instruction = content["instruction"] \
                        if random.random() > self.cond_mask_prob else ""
                    data_dict["input_ids"] = self.tokenizer(
                        instruction,
                        return_tensors="pt",
                        padding="longest",
                        truncation=False,
                    ).input_ids[0]
                
                    # Validate instruction length
                    assert len(data_dict["input_ids"]) <= self.tokenizer_max_length, \
                        f"Instruction length {len(data_dict['input_ids'])} exceeds the maximum length {self.tokenizer_max_length}."
                
                # Convert numpy arrays to PyTorch tensors
                for k, v in data_dict.items():
                    if isinstance(v, np.ndarray):
                        data_dict[k] = torch.from_numpy(v)

                # Final validation - ensure no numpy arrays remain
                for k, v in data_dict.items():
                    assert not isinstance(v, np.ndarray), f"key: {k}, value: {v}"
        
                return data_dict
                
            except BaseException as e:
                # Handle errors with informative logging
                if data_dict is not None:
                    print(f"Error caught when processing sample from {data_dict.get('dataset_name')}:", e)
                else:
                    print(f"Error caught when processing sample:", e)
                traceback.print_exc()
                # Try with a different index
                index = (index + 1) % len(self)


class DataCollatorForVLAConsumerDataset(object):
    """
    Data collator for VLA Consumer Dataset used in supervised training.
    
    This collator handles batching of multi-modal data samples including:
    - Visual observations from multiple cameras
    - Robot state and action sequences
    - Natural language instructions (tokenized or embedded)
    - Point cloud data
    - In-context conditions for goal-oriented tasks
    
    The collator properly handles padding for variable-length sequences
    and ensures all tensors are properly batched and aligned.
    
    Args:
        tokenizer: PreTrainedTokenizer for processing language instructions
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
        """Initialize the data collator with the specified tokenizer."""
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of instances into a single batch dictionary.
        
        This method processes multiple training samples and combines them into
        a single batch with proper padding and tensor stacking. It handles both
        tokenized text inputs and precomputed language embeddings.
        
        Args:
            instances: Sequence of individual data samples (dictionaries)
            
        Returns:
            Dict[str, torch.Tensor]: Batched data ready for model training
        """
        # Initialize batch dictionary with all required keys
        # Each key will collect data from all instances in the batch
        batch = {
            "states": [],              # Robot state sequences
            "actions": [],             # Robot action sequences  
            "state_elem_mask": [],     # Masks indicating valid state elements
            "state_norm": [],          # State normalization parameters
            "images": [],              # Visual observations from cameras
            "data_indices": [],        # Dataset indices for each sample
            "ctrl_freqs": [],          # Control frequencies for each dataset
            "pointclouds": [],         # Point cloud data from depth sensors
            "in_context_conditions": [],  # Concatenated in-context conditions
        }
        
        # Separate lists for language processing (tokenized vs embedded)
        input_ids = []              # For tokenized language instructions
        lang_embeds = []           # For precomputed language embeddings
        lang_embed_lens = []       # Lengths of language embeddings for masking
        
        # Process each instance in the batch
        for instance in instances:
            # Define the basic keys that need tensor conversion and batching
            basic_keys = ['states', 'actions', 'state_elem_mask', 'state_norm', 'in_context_conditions']
            
            # Process basic tensor keys - convert to PyTorch tensors if needed
            for key in basic_keys:
                if key in instance:
                    if isinstance(instance[key], torch.Tensor):
                        item = instance[key]
                    else:
                        # Convert numpy arrays to PyTorch tensors
                        item = torch.from_numpy(instance[key])
                    batch[key].append(item)

            # Handle language data (either tokenized or embedded)
            if "input_ids" in instance:
                # Add tokenized language instructions for on-the-fly processing
                input_ids.append(instance["input_ids"])
            else:
                # Add precomputed language embeddings and track their lengths
                lang_embeds.append(instance["lang_embed"])
                lang_embed_lens.append(instance["lang_embed"].shape[0])
            
            # Process visual and metadata
            batch["images"].append(torch.stack(instance["images"], dim=0))  # Stack camera images
            batch["data_indices"].append(instance["data_idx"])              # Dataset identifier
            batch["ctrl_freqs"].append(instance["ctrl_freq"])               # Control frequency
            batch["pointclouds"].append(instance["pointclouds"])            # 3D point clouds
        
        # Stack tensors along batch dimension for efficient processing
        keys_to_stack = ['states', 'actions', 'state_elem_mask', 'state_norm', "images", "pointclouds", "in_context_conditions"]
        for key in keys_to_stack:
            if key in batch and len(batch[key]) > 0:
                batch[key] = torch.stack(batch[key], dim=0)
        
        # Convert control frequencies to tensor
        batch["ctrl_freqs"] = torch.tensor(batch["ctrl_freqs"])

        # Handle language data with proper padding for variable lengths
        if len(input_ids) > 0:
            # Pad tokenized language instructions to the same length
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            batch["input_ids"] = input_ids
            # Create attention mask to ignore padded tokens
            batch["lang_attn_mask"] = input_ids.ne(self.tokenizer.pad_token_id)
        else:
            # Pad precomputed language embeddings to the same length
            lang_embeds = torch.nn.utils.rnn.pad_sequence(
                lang_embeds,
                batch_first=True,
                padding_value=0)
            # Create attention mask based on original embedding lengths
            input_lang_attn_mask = torch.zeros(
                lang_embeds.shape[0], lang_embeds.shape[1], dtype=torch.bool)
            for i, l in enumerate(lang_embed_lens):
                input_lang_attn_mask[i, :l] = True  # Mark valid positions
            batch["lang_embeds"] = lang_embeds
            batch["lang_attn_mask"] = input_lang_attn_mask
            
            
        return batch


if __name__ == '__main__':
    """
    Unit test to validate the VLAConsumerDataset and DataCollatorForVLAConsumerDataset functionality.
    
    This test creates a dataset instance, loads samples, and validates the data collator.
    """
    import yaml
    from transformers import AutoTokenizer
    from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
    
    print("Starting dataset unit test...")
    
    # Load configuration
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Create mock config for dataset
    dataset_config = config["dataset"]

    # Initialize components
    try:
        # Create tokenizer (using a simple tokenizer for testing)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create image processor
        vision_encoder = SiglipVisionTower(vision_tower="google/siglip-so400m-patch14-384", args=None)
        image_processor = vision_encoder.image_processor
    
        print("✓ Tokenizer and image processor initialized successfully")
        
        # Create dataset
        dataset = VLAConsumerDataset(
            config=dataset_config,
            tokenizer=tokenizer,
            image_processor=image_processor,
            num_cameras=3,
            img_history_size=config['common']['img_history_size'],
            image_size=None,
            auto_adjust_image_brightness=False,
            image_aug=False,
            dataset_type='finetune',
            cond_mask_prob=0.1,
            cam_ext_mask_prob=-1.0,
            state_noise_snr=None,
            use_hdf5=True,
            hdf5_dataset_name="mshab",
            use_precomp_lang_embed=True,
            task_names=[]
        )
        
        print("✓ Dataset created successfully")
        print(f"Dataset length: {len(dataset)}")
        
        # Test getting a single sample
        sample = dataset[0]
        print("✓ Successfully retrieved a sample from dataset")
        
        # Validate sample structure
        expected_keys = [
            'dataset_name', 'data_idx', 'ctrl_freq', 'states', 'actions', 
            'state_elem_mask', 'state_norm', 'in_context_conditions', 
            'images', 'pointclouds', 'lang_embed'
        ]
        
        for key in expected_keys:
            if key in sample:
                print(f"✓ Sample contains key: {key}, shape: {sample[key].shape if hasattr(sample[key], 'shape') else type(sample[key])}")
            else:
                print(f"⚠ Sample missing key: {key}")
        
        # Test data collator
        data_collator = DataCollatorForVLAConsumerDataset(tokenizer=tokenizer)
        
        # Create a batch of samples
        batch_samples = [sample, dataset[1]]
        batch = data_collator(batch_samples)
        
        print("✓ Data collator processed batch successfully")
        
        # Validate batch structure
        expected_batch_keys = [
            'states', 'actions', 'state_elem_mask', 'state_norm', 
            'images', 'pointclouds', 'in_context_conditions', 
            'ctrl_freqs', 'lang_embeds', 'lang_attn_mask'
        ]
        
        for key in expected_batch_keys:
            if key in batch:
                print(f"✓ Batch contains key: {key}, shape: {batch[key].shape if hasattr(batch[key], 'shape') else type(batch[key])}")
            else:
                print(f"⚠ Batch missing key: {key}")
        
        # Validate tensor types and shapes
        assert isinstance(batch['states'], torch.Tensor), "states should be a tensor"
        assert isinstance(batch['actions'], torch.Tensor), "actions should be a tensor"
        assert isinstance(batch['in_context_conditions'], torch.Tensor), "in_context_conditions should be a tensor"
        assert batch['states'].shape[0] == 2, "Batch size should be 2"
        assert batch['actions'].shape[0] == 2, "Batch size should be 2"
        
        print("✓ All tensor type and shape validations passed")
        
        # Test dataset mapping functions
        name2id = dataset.get_dataset_name2id()
        id2name = dataset.get_dataset_id2name()
        
        print(f"✓ Dataset name to ID mapping: {name2id}")
        print(f"✓ Dataset ID to name mapping: {id2name}")
        
        print("\n🎉 All unit tests passed successfully!")
        print("The VLAConsumerDataset and DataCollatorForVLAConsumerDataset are working correctly.")
        
    except Exception as e:
        print(f"❌ Unit test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    print("Unit test completed.")
