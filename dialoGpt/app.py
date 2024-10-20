from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

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
            model = AutoModelForCausalLM.from_pretrained(model_names[model_name])
        else:
            model.load_state_dict(torch.load(model_names[model_name], map_location=device))
        model.to(device)
        current_model_name = model_name


@app.get("/")
def read_root():
    return {"message": "API is running"}


@app.post("/change-model")
def change_model(model_name: str):
    load_model(model_name)
    return {"status": f"Model changed to {model_name}"}


@app.post("/generate-response")
def generate_response(input_text: str):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"])
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
