# Example code for loading and using the model
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info
from PIL import Image

model_name = "Qwen/Qwen2-VL-2B-Instruct"

# Detect device and use the new `dtype` argument (torch_dtype is deprecated)
device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
print(f"Using device: {device}")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)

# Load the processor from Hugging Face Hub
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Add example usage code
image_path = './data/0a1gfi-0.png'
image = Image.open(image_path).convert('RGB')

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": "Convert this image to text"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
# Do NOT force-move inputs when using device_map="auto"; pass CPU tensors and let HF handle placement
# If you explicitly want to run on CPU, the above already returns CPU tensors.

# Inference: Generation of the output
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)
