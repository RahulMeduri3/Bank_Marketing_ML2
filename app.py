# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 19:55:48 2021

@author: rahul
"""

from flask import Flask, render_template, request, Response
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import auc
import io
import pickle

app = Flask(__name__)

df = pd.DataFrame()

rf_model, rf_fpr, rf_tpr, rf_cm = pickle.load(open('rf_model_v1.pkl', 'rb'))
lr_model, lr_fpr, lr_tpr, lr_cm = pickle.load(open('lr_model_v1.pkl', 'rb'))
ada_model, ada_fpr, ada_tpr, ada_cm = pickle.load(open('ada_model_v1.pkl', 'rb'))
xgb_model, xgb_fpr, xgb_tpr, xgb_cm = pickle.load(open('xgb_model_v1.pkl', 'rb'))
vt_model, vt_fpr, vt_tpr, vt_cm = pickle.load(open('vt_model_v1.pkl', 'rb'))


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/getfile', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
       global df
       f = request.files['file']
       df = pd.read_csv(f)
       df = df.drop(columns=df.columns[0], axis=1)
       return Response('Success')

@app.route('/voting')
def voting():
    global df
    my_prediction=vt_model.predict(df)
    my_prediction=my_prediction.tolist()
   
    result_data = pd.DataFrame()
    result_data['Duration'] = df['duration']
    result_data['Age'] = df['age']
    result_data['Balance'] = df['balance']
    result_data['Will_Subscribe'] = my_prediction
    
    return render_template('voting.html', prediction = result_data.values[:10])

@app.route('/plot_voting.png')
def plot_voting_png():
    
    fig = create_voting_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_voting_figure():
    fig = plt.figure()
    idx = np.min(np.where(vt_tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95
    plt.plot(vt_fpr, vt_tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(vt_fpr, vt_tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0,vt_fpr[idx]], [vt_tpr[idx],vt_tpr[idx]], 'k--', color='blue')
    plt.plot([vt_fpr[idx],vt_fpr[idx]], [0,vt_tpr[idx]], 'k--', color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (recall)', fontsize=14)
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    return fig

@app.route('/plot_voting_cm.png')
def plot_voting_cm():
    
    fig = create_voting_cm_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_voting_cm_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    ax = sns.heatmap(vt_cm, annot=True, ax=axis, cmap='Blues')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    return fig

@app.route('/xgBoost')
def xg_boost():
    global df
    my_prediction=xgb_model.predict(df)
    my_prediction=my_prediction.tolist()
   
    result_data = pd.DataFrame()
    result_data['Duration'] = df['duration']
    result_data['Age'] = df['age']
    result_data['Balance'] = df['balance']
    result_data['Will_Subscribe'] = my_prediction
    
    return render_template('xgBoost.html', prediction = result_data.values[:10])

@app.route('/plot_xgb.png')
def plot_xgb_png():
    
    fig = create_xgb_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_xgb_figure():
    fig = plt.figure()
    idx = np.min(np.where(xgb_tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95
    plt.plot(xgb_fpr, xgb_tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(xgb_fpr, xgb_tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0,xgb_fpr[idx]], [xgb_tpr[idx],xgb_tpr[idx]], 'k--', color='blue')
    plt.plot([xgb_fpr[idx],xgb_fpr[idx]], [0,xgb_tpr[idx]], 'k--', color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (recall)', fontsize=14)
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    return fig

@app.route('/plot_xgb_cm.png')
def plot_xgb_cm():
    
    fig = create_xgb_cm_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_xgb_cm_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    ax = sns.heatmap(xgb_cm, annot=True, ax=axis, cmap='Blues')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    return fig

@app.route('/adaBoost')
def ada_boost():
    global df
    my_prediction=ada_model.predict(df)
    my_prediction=my_prediction.tolist()
   
    result_data = pd.DataFrame()
    result_data['Duration'] = df['duration']
    result_data['Age'] = df['age']
    result_data['Balance'] = df['balance']
    result_data['Will_Subscribe'] = my_prediction
    
    return render_template('adaBoost.html', prediction = result_data.values[:10])

@app.route('/plot_ada.png')
def plot_ada_png():
    
    fig = create_ada_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_ada_figure():
    fig = plt.figure()
    idx = np.min(np.where(ada_tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95
    plt.plot(ada_fpr, ada_tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(ada_fpr, ada_tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0,ada_fpr[idx]], [ada_tpr[idx],ada_tpr[idx]], 'k--', color='blue')
    plt.plot([ada_fpr[idx],ada_fpr[idx]], [0,ada_tpr[idx]], 'k--', color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (recall)', fontsize=14)
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    return fig

@app.route('/plot_ada_cm.png')
def plot_ada_cm():
    
    fig = create_ada_cm_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_ada_cm_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    ax = sns.heatmap(ada_cm, annot=True, ax=axis, cmap='Blues')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    return fig

@app.route('/logisticRegression')
def logistic_regression():
    global df
    my_prediction=lr_model.predict(df)
    my_prediction=my_prediction.tolist()
   
    result_data = pd.DataFrame()
    result_data['Duration'] = df['duration']
    result_data['Age'] = df['age']
    result_data['Balance'] = df['balance']
    result_data['Will_Subscribe'] = my_prediction
    
    return render_template('logisticRegression.html', prediction = result_data.values[:10])

@app.route('/plot_lr.png')
def plot_lr_png():
    
    fig = create_lr_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_lr_figure():
    fig = plt.figure()
    idx = np.min(np.where(lr_tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95
    plt.plot(lr_fpr, lr_tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(lr_fpr, lr_tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0,lr_fpr[idx]], [lr_tpr[idx],lr_tpr[idx]], 'k--', color='blue')
    plt.plot([lr_fpr[idx],lr_fpr[idx]], [0,lr_tpr[idx]], 'k--', color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (recall)', fontsize=14)
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    return fig

@app.route('/plot_lr_cm.png')
def plot_lr_cm():
    
    fig = create_lr_cm_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_lr_cm_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    ax = sns.heatmap(lr_cm, annot=True, ax=axis, cmap='Blues')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    return fig

@app.route('/randomForestClassifier')
def random_forest():
    global df
    my_prediction=rf_model.predict(df)
    my_prediction=my_prediction.tolist()
   
    result_data = pd.DataFrame()
    result_data['Duration'] = df['duration']
    result_data['Age'] = df['age']
    result_data['Balance'] = df['balance']
    result_data['Will_Subscribe'] = my_prediction
    
    return render_template('randomForest.html', prediction = result_data.values[:10])

@app.route('/plot.png')
def plot_png():
    
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = plt.figure()
    idx = np.min(np.where(rf_tpr > 0.95))
    plt.plot(rf_fpr, rf_tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(rf_fpr, rf_tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0,rf_fpr[idx]], [rf_tpr[idx],rf_tpr[idx]], 'k--', color='blue')
    plt.plot([rf_fpr[idx],rf_fpr[idx]], [0,rf_tpr[idx]], 'k--', color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (recall)', fontsize=14)
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    return fig

@app.route('/plot_cm.png')
def plot_cm():
    
    fig = create_cm_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_cm_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    ax = sns.heatmap(rf_cm, annot=True, ax=axis, cmap='Blues')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    return fig

if __name__ == '__main__':
	app.run(debug=True, use_reloader=False)