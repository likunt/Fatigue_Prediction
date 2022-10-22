#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
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


# In[2]:


# start the App
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets)
app.title = 'Fatigue Strength Prediction Dashboard'
server = app.server


# In[3]:


# load the ml model
model_deploy = pickle.load(open('../Backend/xgb_model_deploy.pickle', 'rb'))


# In[4]:


prediction_col1 =  dbc.Col([ 
                html.Br(),
                dbc.Row([html.H5(children='Heat Treatment Conditions')]),
    #processing parameters
                dbc.Row([
                    dbc.Col(html.Label(children='Normalizing Temperature (°C) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='NT', type='text', value = '870'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Through Hardening Temperature (°C) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='THT', type='text', value = '855'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Through Hardening Time (minutes) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='THt', type='text', value = '30'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Cooling Rate for Through Hardening (°C/hr) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='THQCr', type='text', value = '8'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Carburization Temperature (°C) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='CT', type='text', value = '30'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Carburization Time (minutes) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='Ct', type='text', value = '0'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Diffusion Temperature (°C) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='DT', type='text', value = '30'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Diffusion Time (minutes) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='Dt', type='text', value = '0'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Quenching Media Temperature for Carburization (°C) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='QmT', type='text', value = '30'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Tempering Temperature (°C) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='TT', type='text', value = '650'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Tempering Time (minutes) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='Tt', type='text', value = '60'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Cooling Rate for Tempering (°C/hr) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='TCr', type='text', value = '24'))
                ]),
    
            ], style = {'padding': '0px 0px 0px 150px'}, width=3)


# In[5]:


prediction_col2 =  dbc.Col([ 
    
                html.Br(),
                dbc.Row([html.H5(children='Chemical Compositions')]),
    
     #chemical composition
                dbc.Row([
                    dbc.Col(html.Label(children='Carbon (C) (wt %) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='C', type='text', value = '0.42'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Silicon (Si) (wt %):'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='Si', type='text', value = '0.29'))
                ]),  
                dbc.Row([
                    dbc.Col(html.Label(children='Manganese (Mn) (wt %):'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='Mn', type='text', value = '0.77'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Phosphorus (P) (wt %):'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='P', type='text', value = '0.015'))
                ]), 
                dbc.Row([
                    dbc.Col(html.Label(children='Sulphur (S) (wt %) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='S', type='text', value = '0.017'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Nickel (Ni) (wt %) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='Ni', type='text', value = '0.12'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Chromium (Cr) (wt %) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='Cr', type='text', value = '1.1'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Copper (Cu) (wt %) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='Cu', type='text', value = '0.09'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Molybdenum (Mo) (wt %) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='Mo', type='text', value = '0.15'))
                ]),
    
                html.Br(),
                dbc.Row([html.H5(children='Upstream Processing Parameters')]),
               
    # reduction ratio
                dbc.Row([
                    dbc.Col(html.Label(children='Reduction Ratio (Ingot to Bar) :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='RedRatio', type='text', value = '500'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Area Proportion of Inclusions Deformed by Plastic Work :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='dA', type='text', value = '0.09'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Area Proportion of Inclusions Occurring in Discontinuous Array :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='dB', type='text', value = '0.01'))
                ]),
                dbc.Row([
                    dbc.Col(html.Label(children='Area Proportion of Isolated Inclusions :'), width={"order": "first"}),
                    dbc.Col(dbc.Input(id='dC', type='text', value = '0.01'))
                ]),
    
            ], style = {'padding': '0px 0px 0px 150px'}, width=4)


# In[ ]:


prediction_col3 =  dbc.Col([ 
    html.Br(), 
    dbc.Row([html.Div(dcc.Graph(id='hist-graph'))], align="Center"),
    
    html.Br(),
   # dbc.Row([dbc.Button('Submit', id='submit-val', n_clicks=0, color="primary", className="d-grid gap-2 col-2 mx-auto")]),
    dbc.Row([dbc.Button('Submit', id='submit-val', n_clicks=0, color="primary", className="d-grid gap-2 col-2 mx-auto")]),
    
    html.Br(),
    dbc.Row([html.Div(id='container-button-basic')], align="Center"),
     
], style = {'padding': '0px 0px 0px 150px'}, width=4)


# In[ ]:


app.layout = html.Div([
    html.H1(children='Fatigue Strength Prediction Dashboard'),

    html.Div(children='''Fatigue Strength Prediction is constructed using chemical composition and processing parameters.'''),
    html.Br(), 
    
    html.Div(
            children=[
            dbc.Label('Class of Steel:', style={'display': 'inline-block', 'margin-right': '15px'}),
            dbc.RadioItems(
                id='class_type',
                options=[
                {'label': 'Carbon and Low Alloy', 'value': 'Carbon_low_alloy'},
                {'label': 'Carburizing', 'value': 'Carburizing'},
                {'label': 'Spring', 'value': 'Spring'}
                ],
            value="Carbon_low_alloy",  
            inline=True),
            ],
        style={'display': 'flex'}
    ),

    
    dbc.Row([
        prediction_col1,
        prediction_col2, 
        prediction_col3,
    ],
    align="start")
    
], style = {'padding': '20px'}) # the end of app.layout


# In[ ]:


# create call back fror prediction
@app.callback(
    Output("container-button-basic", "children"),
    Output('hist-graph', 'figure'),
    # Inputs will trigger your callback; State do not. 
    # If you need the the current “value” - aka State - of other 
    # dash components within your callback, you pass them along via State.
    Input('submit-val', 'n_clicks'),
    State('class_type', 'value'),
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

def update_output(n_clicks, class_type, NT, THT, THt, THQCr, CT, Ct, DT, Dt, QmT, TT, Tt, TCr, C, Si, 
                  Mn, P, S, Ni, Cr, Cu, Mo, RedRatio, dA, dB, dC):
    
    query = pd.DataFrame({'Class': class_type,
                        'NT': float(NT),
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

    p_25 = round(pos-0.67*scale, 3)
    p_50 = round(pos+0., 3)
    p_75 = round(pos+0.67*scale, 3)

    fig = px.histogram(his_df, x = 'Fatigue', histnorm ='percent', nbins = 20, width = 550, height  = 420)
    fig.add_vline(x=p_25, line_width=3, line_dash="dash", line_color="green", annotation_text="25th Percentile: {p_25:,}", annotation_position="top left")
    fig.add_vline(x=p_75, line_width=3, line_dash="dash", line_color="green", annotation_text="75th Percentile: {p_75:,}", annotation_position="top right")
    fig.add_vline(x=p_50, line_width=3, line_dash="dash", line_color="green", annotation_text="50th Percentile (Median): {p_50:,}", annotation_position="top")
    fig.add_vrect(x0=p_25, x1=p_75, line_width=3, fillcolor="red", opacity=0.2)


    return 'The estimated fatigue strength is {p_50:,}. The 50% CI is [{p_25:,}, {p_75:,}].', fig


# In[ ]:


# run the app 
if __name__ == '__main__':
    useport = int(os.environ.get('PORT', 33507))
    app.run_server(debug=False, port=useport)


