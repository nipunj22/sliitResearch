from flask import Flask,request
from flask_cors import CORS , cross_origin

from chat_bot import get_answer
from damage_detect import detect_cost

app = Flask(__name__)
CORS(app , resources={r"/":{"origins":"*"}})

@app.route("/")
def main():
    return "hello world!"

@app.route("/home")
@cross_origin()
def home():
    return "First Page"

@app.route("/v1/chat_bot" , methods=["POST"])
@cross_origin()
def login():
    question = request.json['question']
    try:
        return {
            "state":True,
            "result":get_answer(question)
        }
    except:
        return {
            "state":False,
            "message":"internal error"
        }

@app.route("/v1/damage_cost" , methods=["POST"])
@cross_origin()
def login2():
    uploaded_file = request.files['image']
    body = request.form
    try:
        uploaded_file.save('uploaded.png')
        return {
            "state":True,
            "result":detect_cost(body=body)
        }
    except:
        return {
            "state":False,
            "message":"internal error"
        }

if __name__ == '__main__':
    app.debug = True
    app.run(host='localhost',port=5000)
