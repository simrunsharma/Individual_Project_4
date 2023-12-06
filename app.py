from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        user_input = request.form['input_text']
        inputs = tokenizer.encode(user_input, return_tensors='pt')
        summary_ids = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return render_template('result.html', input_text=user_input, generated_text=output)

if __name__ == '__main__':
    app.run(debug=True)