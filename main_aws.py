from transformers import AutoTokenizer
from flask import Flask, jsonify, request  # import objects from the Flask model
from sagemaker.huggingface import HuggingFaceModel
import sagemaker

app = Flask(__name__)  # define app using Flask
# IAM role with permissions to create endpoint
role = "ltts-gptj"

# public S3 URI to gpt-j artifact
model_uri="s3://gptj-sagemaker-inference/gpt-j/model.tar.gz"

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    model_data=model_uri,
    transformers_version='4.12.3',
    pytorch_version='1.9.1',
    py_version='py38',
    role=role, 
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
    initial_instance_count=1, # number of instances
    instance_type='ml.g4dn.xlarge' #'ml.p3.2xlarge' # ec2 instance type
)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
def predict(context,temperature,max_length,end_sequence,return_full_text):
    return predictor.predict({'inputs': context,
      "parameters" : {
        "max_length": int(len(context) + max_length),
        "temperature": float(temperature),
        "eos_token_id": int(tokenizer.convert_tokens_to_ids(end_sequence)),
        "return_full_text":return_full_text
      }
    })
@app.route('/complete', methods=['POST'])
def generate_text():
    context = request.json['context']
    temperature = request.json['temperature']
    max_length = request.json['max_length']
    end_sequence = request.json['end_sequence']
    return_full_text = request.json['return_full_text']
    print(context)
    res = predict(context,temperature,max_length,end_sequence,return_full_text)
    return jsonify({'output': res})

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=False, port=8080)
