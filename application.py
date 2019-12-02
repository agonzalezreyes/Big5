from flask import Flask, render_template, request, redirect
from application import db
from application.models import Data
from application.forms import EnterDBInfo, RetrieveDBInfo
from application import twit
from application import big5
import json

# initalization
application = Flask(__name__)
application.debug = False
application.secret_key = 'KEY HERE'

@application.before_request
def before_request():
    if request.url.startswith('http://'):
        return redirect(request.url.replace('http://', 'https://'), code=301)

@application.route('/.well-known/acme-challenge/<challenge>')
def letsencrypt_check(challenge):
    challenge_response = {
    "<challenge 1 here>":"<challenge 1 result here>",
    "<challenge 2 here>":"<challenge 2 result here>"
    }
    return Response(challenge_response[challenge], mimetype='text/plain')

@application.route('/', methods=['GET', 'POST'])
@application.route('/index', methods=['GET', 'POST'])
def index():
    form1 = EnterDBInfo(request.form)
    form2 = RetrieveDBInfo(request.form)

    if request.method == 'POST' and form1.validate() and twit.verify(form1.dbNotes.data):
        user = twit.clean(form1.dbNotes.data)
        data_entered = Data(notes=user)
        try:
            db.session.add(data_entered)
            db.session.commit()
            db.session.close()
        except:
            db.session.rollback()
        big5.train_models()
        tweets = twit.get_tweets(user)
        result = big5.check_personality(tweets)
        result['user'] = user
        data_json = json.dumps(result)
        loaded_data = json.loads(data_json)
        return render_template('results.html', notes=loaded_data)
    return render_template('index.html', form1=form1, form2=form2)

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=443, threaded=True, ssl_context=('/etc/letsencrypt/live/big5project.alejandrina.me/fullchain.pem', '/etc/letsencrypt/live/big5project.alejandrina.me/privkey.pem'))
