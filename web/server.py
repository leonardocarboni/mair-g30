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
        session['id'] = secrets.token_hex()
        session['name'] = request.form.get("name")
        session['email'] = request.form.get("email")
        session['useDT'] = request.form.get("useDT") == "on"
        session['useCL'] = request.form.get("useCL") == "on"
        session['useAC'] = request.form.get("useAC") == "on"
        session['informations'] = {'food': None, 'area': None,
                                   'price': None, 'suitable_list': None, 'extra': None}
        session.modified = True
        return render_template('main.html', title='RestaurantBot')
    elif session.get("id") != None:
        return render_template('main.html', title='RestaurantBot')
    else:
        return render_template('welcome.html', title='Register')


@app.route('/message', methods=['POST'])
def message():
    if request.method == 'POST':
        message = bot.get_response(request.get_json()['msg'])
        return jsonify({'message': message})
