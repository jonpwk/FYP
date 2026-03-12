from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
import torch
from PIL import Image
from datasets import load_dataset
import pandas as pd
from io import BytesIO

prompt = "Transcribe the Jawi script in this image into Jawi text"

# Load config and fix missing rope_scaling
config = AutoConfig.from_pretrained("culturalheritagenus/Jawi-OCR-Qwen-v2")
config.rope_scaling = {'type': 'default', 'mrope_section': [16, 24, 24], 'rope_type': 'default'}

model = AutoModelForImageTextToText.from_pretrained("culturalheritagenus/Jawi-OCR-Qwen-v2", config=config)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

def process_image(image):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        print(f"Input IDs Length: {inputs['input_ids'].shape[1]}")

        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )

        print(f"Generated IDs Length: {generated_ids.shape[1]}")

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        generated_text = processor.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        return generated_text

    except Exception as e:
        print(f"Error during generation: {e}")
        return "Error: Could not process image"

# Load the Parquet file
df = pd.read_parquet('./data_v4/test')

for idx, row in df.iterrows():
    # If your image is stored as a dict with a 'bytes' key:
    # Get image bytes
    image_data = row['Image']
    image_bytes = image_data['bytes']
    # Convert bytes to PIL Image
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    # Get ground truth
    ground_truth = row['Text']
    print(f"Ground Truth: {ground_truth}")
    print(f"Image Bytes Length: {len(image_bytes)}")
    result = process_image(image)
    print(f"Sample {idx}: {result}")
    # Optionally break after one for testing
    break
