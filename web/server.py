from flask_bootstrap import Bootstrap5
from flask import Flask
from flask import render_template, request, jsonify, session
import secrets
import bot

app = Flask(__name__)
app.secret_key = secrets.token_hex()

bootstrap = Bootstrap5(app)

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['name'] = request.form.get("name")
        session['email'] = request.form.get("email")
        session['useDT'] = request.form.get("useDT")
        session['useCL'] = request.form.get("useCL")
        session['useAC'] = request.form.get("useAC")
        bot.initialize(session['useDT'], session['useCL'], session['useAC'])
        return render_template('main.html', title='Test2')
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
