import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the shared tokenizer (using a tokenizer from DialoGPT models)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Define the model names, including the locally saved fine-tuned model
model_names = {
    "DialoGPT-med-FT": "DialoGPT-med-FT.bin",
    "DialoGPT-medium": "microsoft/DialoGPT-medium"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the default model initially
current_model_name = "DialoGPT-med-FT"
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
model.load_state_dict(torch.load(model_names[current_model_name], map_location=device))
model.to(device)

def load_model(model_name):
    global model, current_model_name
    if model_name != current_model_name:
        # Load the new model and update the current model reference
        if model_name == "DialoGPT-medium":
            model = AutoModelForCausalLM.from_pretrained(model_names[model_name]).to(device)
        elif model_name == "DialoGPT-med-FT":
            model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
            model.load_state_dict(torch.load(model_names[model_name], map_location=device))
            model.to(device)
        current_model_name = model_name

def respond(
    message,
    history: list[dict],
    model_choice,
    max_tokens,
    temperature,
    top_p,
):
    # Load the selected model if it's different from the current one
    load_model(model_choice)

    # Prepare the input by concatenating the history into a dialogue format
    input_text = ""
    for message_pair in history:
        input_text += f"{message_pair['role']}: {message_pair['content']}\n"
    input_text += f"User: {message}\nAssistant:"

    # Tokenize the input text using the shared tokenizer
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)

    # Generate the response using the selected DialoGPT model
    output_tokens = model.generate(
        inputs["input_ids"].to(device),
        max_length=len(inputs["input_ids"][0]) + max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )

    # Decode and return the assistant's response
    response = tokenizer.decode(output_tokens[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    yield response

# Define the Gradio interface
demo = gr.ChatInterface(
    respond,
    type='messages',
    additional_inputs=[
        gr.Dropdown(choices=["DialoGPT-med-FT", "DialoGPT-medium"], value="DialoGPT-med-FT", label="Model"),
        gr.Slider(minimum=1, maximum=100, value=15, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
    ],
)

if __name__ == "__main__":
    demo.launch()