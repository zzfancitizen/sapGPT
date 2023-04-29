from src import AskMe
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
CORS(app)
bot = AskMe()


@app.route('/ask', methods=['POST'])
def process_data():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data found'}), 400

    response = {
        'status': 'success',
        'message': 'Data processed successfully',
        'data': bot.ask(data['question'])
    }

    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True)
