from flask import Flask, render_template, request, jsonify
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

app = Flask(__name__)

# Load model
model = PegasusForConditionalGeneration.from_pretrained("model")
tokenizer = PegasusTokenizer.from_pretrained('tokenizer')

def get_response(input_text, num_return_sequences=1, num_beams=10, max_length=500, temperature=1.5):
    try:
        batch = tokenizer([input_text], truncation=True, padding='longest', max_length=max_length, return_tensors="pt")
        translated = model.generate(
            **batch,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text[0]
    except IndexError as e:
        print(f"Error: {e}")
        return "An error occurred during text generation."


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/paraphrase', methods=['POST'])
def predict():
    data = request.get_json()  # Get the JSON data from the request
    input_text = data.get('input-text')
    if input_text:
        paraphrase = get_response(input_text, num_return_sequences=1, num_beams=10)
        return jsonify({'paraphrase': paraphrase})  # Return the paraphrase as JSON
    return jsonify({'paraphrase': 'No input text provided.'})  # Return error message if no input provided

if __name__ == "__main__":
    app.run(debug=True, port=5004)
