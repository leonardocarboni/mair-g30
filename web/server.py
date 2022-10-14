from flask_bootstrap import Bootstrap5
from flask import Flask
from flask import render_template, request, jsonify, session
import secrets
import bot

app = Flask(__name__)
app.secret_key = secrets.token_hex()

bootstrap = Bootstrap5(app)

bot.initialize(1, 1, 1)


@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['name'] = request.form.get("name")
        session['email'] = request.form.get("email")
        session['useDT'] = request.form.get("useDT")
        #bot.initialize(session['useDT'], session['useDT'], session['useDT'])
        return render_template('main.html', title='Test')
    elif session.get("name") != None:
        return render_template('main.html', title='Test')
    else:
        return render_template('welcome.html', title='Register')


@app.route('/message', methods=['POST'])
def message():
    if request.method == 'POST':
        message = bot.get_response(request.get_json()['msg'])
        return jsonify({'message': message})


if __name__ == '__main__':
    app.run(port=5001, debug=True)
