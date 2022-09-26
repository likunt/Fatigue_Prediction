#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

import pickle
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# start the App
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets)
app.title = 'Fatigue Strength Prediction Dashboard'
server = app.server


# load the ml model
model_deploy = pickle.load(open('../Backend/xgb_model_deploy.pickle', 'rb'))


prediction_col1 =  dbc.Col([ 
                html.Br(),
                dbc.Row([html.H3(children='Predict Fatigue Strength')]),
    #processing parameters
                dbc.Row([
                    dbc.Col(html.Label(children='Normalizing Temperature (°C) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='NT', type='text', value = '870', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Through Hardening Temperature (°C) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='THT', type='text', value = '855', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Through Hardening Time (minutes) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='THt', type='text', value = '30', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Cooling Rate for Through Hardening (°C/hr) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='THQCr', type='text', value = '8', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Carburization Temperature (°C) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='CT', type='text', value = '30', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Carburization Time (minutes) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='Ct', type='text', value = '0', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Diffusion Temperature (°C) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='DT', type='text', value = '30', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Diffusion Time (minutes) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='Dt', type='text', value = '0', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Quenching Media Temperature for Carburization (°C) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='QmT', type='text', value = '30', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Tempering Temperature (°C) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='TT', type='text', value = '650', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Tempering Time (minutes) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='Tt', type='text', value = '60', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Cooling Rate for Tempering (°C/hr) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='TCr', type='text', value = '24', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
              
    
    #chemical composition
                dbc.Row([
                    dbc.Col(html.Label(children='Carbon (C) (wt %) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='C', type='text', value = '0.42', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Silicon (Si) (wt %):'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='Si', type='text', value = '0.29', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),  
                dbc.Row([
                    dbc.Col(html.Label(children='Manganese (Mn) (wt %):'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='Mn', type='text', value = '0.77', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Phosphorus (P) (wt %):'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='P', type='text', value = '0.015', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]), 
                dbc.Row([
                    dbc.Col(html.Label(children='Sulphur (S) (wt %) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='S', type='text', value = '0.017', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Nickel (Ni) (wt %) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='Ni', type='text', value = '0.12', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Chromium (Cr) (wt %) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='Cr', type='text', value = '1.1', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Copper (Cu) (wt %) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='Cu', type='text', value = '0.09', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Molybdenum (Mo) (wt %) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='Mo', type='text', value = '0.15', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
    
    # reduction ratio
                dbc.Row([
                    dbc.Col(html.Label(children='Reduction Ratio (Ingot to Bar) :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='RedRatio', type='text', value = '500', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Area Proportion of Inclusions Deformed by Plastic Work :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='dA', type='text', value = '0.09', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Area Proportion of Inclusions Occurring in Discontinuous Array :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='dB', type='text', value = '0.01', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Area Proportion of Isolated Inclusions :'), width={"order": "first"}, style = {'padding': '15px 0px 0px 0px'}),
                    dbc.Col(dbc.Input(id='dC', type='text', value = '0.01', style = {'padding': '5px 0px 5px 10px', 'width': '200px'}))
                ]),
    
                html.Br(),
                dbc.Row([dbc.Button('Submit', id='submit-val', n_clicks=0, color="primary")]),
                html.Br(),
                dbc.Row([html.Div(id='container-button-basic')])
            ], style = {'padding': '0px 0px 0px 150px'})

prediction_col2 =  dbc.Col([ html.Br(), html.Div(dcc.Graph(id='hist-graph'))], style = {'padding': '0px 0px 0px 0px'})


# In[6]:


# prepare the layout
app.layout = html.Div([
    html.H1(children='Fatigue Strength Prediction Dashboard'),

    html.Div(children='''Fatigue Strength Prediction is constructed using chemical composition and processing parameters.'''),
    html.Div(children='Developer: Likun Tan'),
    html.Br(),
    
    dcc.Tab(label='Fatigue Strength Prediction', children = [
        dbc.Row([prediction_col1, prediction_col2])
       

    ]) # end of all tabs

], style = {'padding': '20px'}) # the end of app.layout


# In[7]:


# create call back fror prediction
@app.callback(
    Output("container-button-basic", "children"),
    Output('hist-graph', 'figure'),
    # Inputs will trigger your callback; State do not. 
    # If you need the the current “value” - aka State - of other 
    # dash components within your callback, you pass them along via State.
    Input('submit-val', 'n_clicks'),
    State('NT', 'value'),
    State('THT', 'value'),
    State('THt', 'value'), 
    State('THQCr', 'value'),
    State('CT', 'value'),
    State('Ct', 'value'),
    State('DT', 'value'), 
    State('Dt', 'value'),
    State('QmT', 'value'),
    State('TT', 'value'),
    State('Tt', 'value'), 
    State('TCr', 'value'),
    State('C', 'value'),
    State('Si', 'value'),
    State('Mn', 'value'), 
    State('P', 'value'),
    State('S', 'value'),
    State('Ni', 'value'),
    State('Cr', 'value'), 
    State('Cu', 'value'),
    State('Mo', 'value'),
    State('RedRatio', 'value'),
    State('dA', 'value'), 
    State('dB', 'value'),
    State('dC', 'value')
)

def update_output(n_clicks, NT, THT, THt, THQCr, CT, Ct, DT, Dt, QmT, TT, Tt, TCr, C, Si, 
                  Mn, P, S, Ni, Cr, Cu, Mo, RedRatio, dA, dB, dC):
    
    query = pd.DataFrame({'NT': float(NT),
                        'THT':THT,
                        'THt': float(THt),
                        'THQCr': float(THQCr),
                        'CT': float(CT),
                        'Ct': float(Ct),
                        'DT':DT,
                        'Dt': float(Dt),
                        'QmT': float(QmT),
                        'TT': float(TT),
                        'Tt': float(Tt),
                        'TCr':TCr,
                        'C': float(C),
                        'Si': float(Si),
                        'Mn': float(Mn),
                        'P': float(P),
                        'S': float(S),
                        'Ni': float(Ni),
                        'Cr': float(Cr),
                        'Cu': float(Cu),
                        'Mo': float(Mo),
                        'RedRatio': float(RedRatio),
                        'dA': float(dA),
                        'dB': float(dB),
                        'dC': float(dC)
                        }, index=[0])
    
    prediction = model_deploy.predict(query)[0]
    output = prediction

    pos = prediction
    scale = model_deploy.st_dev
    size = 200
    np.random.seed(123)
    values = np.random.normal(pos, scale, size)
    his_df = pd.DataFrame(values, columns = ['Fatigue'])

    p_25 = pos-0.67*scale
    p_50 = pos
    p_75 = pos+0.67*scale

    fig = px.histogram(his_df, x = 'Fatigue', histnorm ='percent', nbins = 20, width = 900, height  = 600)
    fig.add_vline(x=p_25, line_width=3, line_dash="dash", line_color="green", annotation_text=f"25th Percentile: {p_25:,}", annotation_position="top left")
    fig.add_vline(x=p_75, line_width=3, line_dash="dash", line_color="green", annotation_text=f"75th Percentile: {p_75:,}", annotation_position="top right")
    fig.add_vline(x=p_50, line_width=3, line_dash="dash", line_color="green", annotation_text=f"50th Percentile (Median): {p_50:,}", annotation_position="top")
    fig.add_vrect(x0=p_25, x1=p_75, line_width=3, fillcolor="red", opacity=0.2)


    return f'The estimated fatigue strength is {p_50:,}. The 50% CI is [{p_25:,}, {p_75:,}].', fig


# run the app 
if __name__ == '__main__':
    app.run_server(debug=False, port=8000)




