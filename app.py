#!/usr/bin/env python
# coding: utf-8


from flask import Flask , render_template , request

app = Flask(__name__)


# @app.route will find the mentioned word in browser url......whichever found, that will be displayed
'''
@app.route('/galary')
def gal():
    return render_template('home_1.html')


@app.route('/')
def bas():
    return('welcome to home')

@app.route('/cart')
def cart():
    return('welcome to cart page')

@app.route('/contact')
def con():
    return('wlecome to contact page')


@app.route('/predict',methods=['post'])
def predict():
    exp = request.form.get('experience') #name should be same as in html code
    phone = request.form.get('phone number')
    mail = request.form.get('email')

    print(exp)
    print(phone)
    print(mail)

    return('got the values')
'''

import joblib

@app.route('/')
def base():
     return render_template('home.html')

@app.route('/prediction', methods=['post'])
def prediction():
   
    model = joblib.load('diabetic_80.pkl')

    preg = request.form.get('preg')
    plas = request.form.get('plas')
    pres = request.form.get('pres')
    skin = request.form.get('skin')
    test = request.form.get('test')
    mass = request.form.get('mass')
    pedi = request.form.get('pedi')
    age = request.form.get('age')


    print(preg , plas , pres , skin , test ,  mass , pedi , age)
    output = model.predict([[preg , plas , pres , skin , test ,  mass , pedi , age]])

    if output[0] == 0:
      data = 'person in not diabetic'
    else:
      data = 'person is diabetic'

    return render_template('home_1.html', data = data)
    


if __name__ == "__main__":
    app.run(debug=True)
#app will run only if main module is running





