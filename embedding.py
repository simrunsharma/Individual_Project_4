from flask import Flask, request, render_template
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

embedding = Flask(__name__)

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
tok = BartTokenizer.from_pretrained("facebook/bart-large")

@embedding.route("/")
def index():
    return render_template("index.html")

@embedding.route("/embed", methods=["POST"])
def embed():
    if request.method == "POST":
        data = request.form["data"]
        batch = tok(data, return_tensors="pt")
        result = model.generate(batch["input_ids"], max_new_tokens=15)
        completion = tok.batch_decode(result, skip_special_tokens=True)
        return render_template("result.html", data=data, completion=completion[0])

if __name__ == "__main__":
    embedding.run(debug=True)