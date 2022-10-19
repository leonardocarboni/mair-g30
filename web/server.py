from flask_bootstrap import Bootstrap5
from flask import Flask
from flask import render_template, request, jsonify, session, redirect
import secrets
import bot
import utils

app = Flask(__name__)
app.secret_key = secrets.token_hex()

bootstrap = Bootstrap5(app)


@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['id'] = secrets.token_hex(9)
        session['name'] = request.form.get("name")
        session['email'] = request.form.get("email")
        session['useML'] = request.form.get("useML") == "on"
        session['useCL'] = request.form.get("useCL") == "on"
        session['useAC'] = request.form.get("useAC") == "on"
        session['state'] = 1 #welcome
        utils.begin_file()
        session['informations'] = {'food': None, 'area': None,
                                   'price': None, 'suitable_list': None, 'extra': None, 'attempt': 0}
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
        utils.save_msg_on_file(message, True)
        return jsonify({'message': message})

@app.route('/reset')
def reset():
    session.clear()
    return redirect('/')