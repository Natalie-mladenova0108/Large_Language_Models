import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Concatenate the messages into a single string
system_message = "You are a pirate chatbot who always responds in pirate speak!"
user_message = "Who are you?"

# Combine the messages into one prompt
prompt = f"{system_message}\nUser: {user_message}\nAssistant:"

# Generate the response
outputs = pipe(
    prompt,
    max_new_tokens=256,
)

# Print the generated response
print(outputs[0]["generated_text"])
