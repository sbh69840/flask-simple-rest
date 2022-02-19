from flask import Flask, jsonify, request  # import objects from the Flask model
from transformers import AutoModelForCausalLM, AutoTokenizer
app = Flask(__name__)  # define app using Flask

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

def predict(context,temperature,max_length):
    input_ids = tokenizer(context, return_tensors="pt").input_ids
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=temperature,
        max_length=max_length,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)
    print(len(gen_text))
    return gen_text[0]
@app.route('/complete', methods=['POST'])
def generate_text():
    context = request.json['context']
    temperature = request.json['temperature']
    max_length = request.json['max_length']
    res = predict(context,temperature,max_length)
    return jsonify({'output': res})

if __name__ == '__main__':
    app.run(debug=False, port=5000)  # run app on port 8080 in debug mode
