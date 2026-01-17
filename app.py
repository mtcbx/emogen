from flask import Flask, render_template, request, redirect, url_for
from diffusers import StableDiffusionPipeline
import os
import torch

app = Flask(__name__)

output_dir = "static/generated"
os.makedirs(output_dir, exist_ok=True)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()
dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

if device == "mps":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

model_path = "fine_tuned_model"
pipeline = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=dtype
)
pipeline.safety_checker = None

pipeline = pipeline.to(device)
print(f"Using device={device} dtype={dtype}")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form.get("prompt")
        
        num_steps = 50
        guidance = 7.5

        image = pipeline(prompt, num_inference_steps=num_steps, guidance_scale=guidance).images[0]

        output_path = os.path.join(output_dir, "generated_image.png")
        image.save(output_path)

        return redirect(url_for("index", image="generated_image.png", prompt=prompt))

    image_path = request.args.get("image")
    prompt = request.args.get("prompt", "")
    return render_template("index.html", image_path=image_path, prompt=prompt)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
