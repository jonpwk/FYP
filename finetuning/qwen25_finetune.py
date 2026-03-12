"""
Fixed fine-tuning script for Qwen2.5-VL models.
Simplified collate function to avoid hanging issues.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Updated imports for Qwen2.5-VL
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
from datasets import load_dataset

from PIL import Image
import base64
from io import BytesIO

from functools import partial
from tqdm import tqdm


def find_assistant_content_sublist_indexes(l):
    """
    Find the start and end indexes of assistant content sublists within a given list.
    Updated for Qwen2.5-VL tokenization.
    """
    start_indexes = []
    end_indexes = []

    # Qwen2.5-VL uses similar token IDs to Qwen2-VL
    assistant_start_token = 151644  
    assistant_end_token = 151645    

    for i in range(len(l) - 1):
        if l[i] == assistant_start_token and l[i + 1] == 77091:
            start_indexes.append(i)
            for j in range(i + 2, len(l)):
                if l[j] == assistant_end_token:
                    end_indexes.append(j)
                    break

    return list(zip(start_indexes, end_indexes))


class HuggingFaceDataset(Dataset):
    """
    Updated Dataset class for Qwen2.5-VL models.
    """
    def __init__(self, dataset, image_column, text_column, user_text="Convert this image to text"):
        self.dataset = dataset
        self.image_column = image_column
        self.text_column = text_column
        self.user_text = user_text

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_column]
        assistant_text = item[self.text_column]

        # Ensure image is properly formatted
        if isinstance(image, str):
            if image.startswith('data:image'):
                # Handle base64 encoded images
                image_data = image.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
            else:
                # Handle file paths
                image = Image.open(image)
        
        # Convert to RGB if necessary
        if hasattr(image, 'mode') and image.mode != 'RGB':
            image = image.convert('RGB')

        # Updated format for Qwen2.5-VL
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.user_text}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": str(assistant_text)}
                    ]
                }
            ]
        }


def collate_fn(batch, processor, device):
    """
    Simplified collate function for Qwen2.5-VL models using chat template.
    This version is much simpler and should avoid hanging issues.
    """
    messages_list = [item["messages"] for item in batch]
    
    # Apply chat template to get text format
    texts = []
    images = []
    
    for messages in messages_list:
        # Apply chat template to get formatted text
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)
        
        # Extract image from messages
        image = messages[0]["content"][0]["image"]
        # Ensure image is PIL Image
        if not isinstance(image, Image.Image):
            if isinstance(image, str):
                if image.startswith('data:image'):
                    image_data = image.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(BytesIO(image_bytes))
                else:
                    image = Image.open(image)
            
        if hasattr(image, 'mode') and image.mode != 'RGB':
            image = image.convert('RGB')
            
        images.append(image)
    
    # Process with the processor (similar to Qwen2-VL approach)
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    
    # Move to device
    inputs = inputs.to(device)
    
    # Create labels
    input_ids_lists = inputs['input_ids'].tolist()
    labels_list = []
    
    for ids_list in input_ids_lists:
        # Initialize with -100 (ignored in loss)
        label_ids = [-100] * len(ids_list)
        
        # Find assistant content and set labels
        for begin_idx, end_idx in find_assistant_content_sublist_indexes(ids_list):
            # Set labels for assistant content (skip first 2 tokens)
            label_ids[begin_idx+2:end_idx+1] = ids_list[begin_idx+2:end_idx+1]
        
        labels_list.append(label_ids)
    
    # Convert to tensor
    labels = torch.tensor(labels_list, dtype=torch.long)
    labels = labels.to(device)
    
    return inputs, labels


def validate(model, val_loader):
    """
    Validate the model on the validation dataset.
    """
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            inputs, labels = batch
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    model.train()
    return avg_val_loss


def train_and_validate(
    model_name,
    output_dir,
    dataset_name,
    image_column,
    text_column,
    device="cuda",
    user_text="Transcribe the Jawi script in this image into Jawi text",
    min_pixel=256,
    max_pixel=384,
    image_factor=28,
    num_accumulation_steps=2,
    eval_steps=10000,
    max_steps=100000,
    train_select_start=0,
    train_select_end=55,
    val_select_start=0,
    val_select_end=14,
    train_batch_size=4,
    val_batch_size=1,
    train_field="train",
    val_field="validation"
):
    """
    Updated training and validation function for Qwen2.5-VL models.
    """
    # Load Qwen2.5-VL model and processor
    print("Loading config!")
    config = AutoConfig.from_pretrained(model_name)
    config.rope_scaling = {'type': 'default', 'mrope_section': [16, 24, 24], 'rope_type': 'default'}

    print(f"Loading model: {model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, 
        config=config,
    )
    print("Model loaded successfully!")
    
    processor = AutoProcessor.from_pretrained(
        model_name
    )
    print("Processor loaded successfully!")
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Create training and validation datasets
    train_dataset = HuggingFaceDataset(
        dataset[train_field].select(range(train_select_start, train_select_end)),
        image_column,
        text_column,
        user_text
    )
    
    val_dataset = HuggingFaceDataset(
        dataset[val_field].select(range(val_select_start, val_select_end)),
        image_column,
        text_column,
        user_text
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, processor=processor, device=device)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, processor=processor, device=device)
    )
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    # Training loop
    model.train()
    global_step = 0
    progress_bar = tqdm(total=max_steps, desc="Training")
    
    print("Starting training loop...")
    
    while global_step < max_steps:
        for batch in train_loader:
            global_step += 1
            inputs, labels = batch
            
            # Forward pass
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / num_accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if global_step % num_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item() * num_accumulation_steps})
            
            # Perform evaluation and save model every eval_steps
            if global_step % eval_steps == 0 or global_step == max_steps:
                print(f"\nEvaluating at step {global_step}...")
                avg_val_loss = validate(model, val_loader)
                print(f"Validation loss: {avg_val_loss:.4f}")
                
                # Save the model and processor
                save_dir = os.path.join(output_dir, f"model_step_{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                processor.save_pretrained(save_dir)
                print(f"Model saved to {save_dir}")
                
                model.train()  # Set the model back to training mode
            
            if global_step >= max_steps:
                save_dir = os.path.join(output_dir, f"final")
                model.save_pretrained(save_dir)
                processor.save_pretrained(save_dir)
                break
        
        if global_step >= max_steps:
            save_dir = os.path.join(output_dir, f"final")
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            break
    
    progress_bar.close()
    print("Training completed!")


def inference_qwen25(model_path, image_path, prompt="Convert this image to text"):
    """
    Inference function for Qwen2.5-VL models using chat template.
    """
    # Load the fine-tuned model and processor
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Load and process the image
    image = Image.open(image_path).convert('RGB')
    
    # Create the conversation format for Qwen2.5-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Process inputs using chat template
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    )
    
    # Move to device
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode the response
    response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    return response


def batch_inference_qwen25(model_path, image_paths, prompt="Convert this image to text"):
    """
    Batch inference function for Qwen2.5-VL models using chat template.
    """
    # Load the fine-tuned model and processor
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    results = []
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Load and process the image
            image = Image.open(image_path).convert('RGB')
            
            # Create the conversation format for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process inputs using chat template
            text = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            )
            
            # Move to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
            
            # Decode the response
            response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            
            results.append({
                "image_path": image_path,
                "prediction": response,
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "image_path": image_path,
                "prediction": "",
                "status": f"error: {str(e)}"
            })
    
    return results
