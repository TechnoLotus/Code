#%%
import os
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from datetime import datetime

data = pd.read_csv('datadashboard.csv', sep='|', index_col=0)
budget = [1000000, 4000000]
roominput = 3
parkinginput = 1
data['neighbourhood'] = ['Pinheiros' if pinheiros == 1 else 'Vila Madalena' if vilamada == 1 else 'Butantan' for pinheiros, vilamada in zip(data['pinheiros'],data['vilamadalena'])]



#%%



mapbox = px.scatter_mapbox(data[(data['price'].between(budget[0],budget[1]))&
                    (data['rooms']>roominput)&
                    (data['parking']>parkinginput)&
                    (data['price']/data['pred']<0.8)],
                    lat='latitude', lon='longitude',
                    color='neighbourhood', 
                    #color_discrete_map={'Pinheiros': 'rgba(128,0,128,0.5)', 'Vila Madalena': 'rgba(255,0,0,0.5)', 'Butantan': 'rgba(0,128,0,0.5)'},
                    hover_data=['price', 'neighbourhood', 'link'],
                    zoom = 11,
                    mapbox_style='open-street-map')


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


server = app.server

app.title = 'NCI Real Estate'

app.layout = html.Div(children=[
    html.Div([dcc.Markdown('''
    ***Dashboard prepared by:***  
    **Name:** Marcio Moraes  
    **Student Number:** 18182691
    ''',
    className = 'nine columns',        
    style={'margin-left': '0', 'textAlign': 'left', 'float': 'left'}),
    
    html.Img(
        src='https://raw.githubusercontent.com/TechnoLotus/Code/master/Images/eu.jpg',
        className='three columns',
        style={
            'height': '5%',
            'width': '5%',
            'float': 'right',
            'position': 'relative',
            'border-radius': '50%'}
    )
    ], className='row'),


    
    dcc.Tabs([
        dcc.Tab(label='Find Bargain Properties', children=[html.Div([
            dcc.Markdown('''
            #### Top Bargains - 20\% off Predicted value!
            Select your budget, number of rooms and number of parking spaces to choose your next apartment
            You can click the neighbourhood name to show/hide the apartments in the area  
            '''),

            html.Div(html.Div([
                dcc.Graph(
                    id='mapbox',
                    figure=mapbox
            ),

            html.P('Select your budget'),
            dcc.RangeSlider(
            id='budget-input',
            min=0,
            max=10000000,
            step=250000,
            marks={i: f'{i/1000:,.0f}k' for i in sorted({*range(0,1000000,250000),*range(1000000,11000000,1000000)})},
            tooltip={'placement': 'top'},
            value=[1000000, 4000000]),


            ], className='ten columns'), className='row'),
            
            html.Br(),

            html.Div([html.Div([html.P('Choose the minimum number of bedrooms:'),
            dcc.Dropdown(
                id='room-input',
                options=[{'label': f'{i:.0f}', 'value': i} for i in sorted(data.rooms.unique())],
                value=1,
                clearable=False,
                className='three columns'
            ),
            ], className='five columns'),

        
            html.Div([html.P('Choose the minimum number of parking spaces:'),
            dcc.Dropdown(
                id='parking-input',
                options=[{'label': f'{i:.0f}', 'value': i} for i in sorted(data.parking.unique())],
                value=1,
                clearable=False,
                className='three columns'
            ),
            ], className='five columns')
        ], className='row')
        ],style={'contentAlign': 'right'})]),

        dcc.Tab(label='Predict the value of your property', children=[
            html.Div([
                html.H1('Check how much your property values today:'),
                
                html.Div(['Area: ',
                        dcc.Input(id='area-input', placeholder=f'Input the area in m\N{SUPERSCRIPT TWO}', type='number')]),
                html.Br(),

                html.Div(['Choose your region:',
                dcc.RadioItems(
                    id='region-input',
                    options=[{'label': i, 'value': i} for i in sorted(data.neighbourhood.unique())],
                    value='Pinheiros',
                    labelStyle={'display': 'inline-block'}
                )]),
                html.Br(),

                html.Button(id='submit-button', n_clicks=0, children='Submit'),
                html.Br(),
                
                html.H1(id='output-pred')])
        ])
    ])
])



@app.callback(
    Output('mapbox', 'figure'),
    [Input('budget-input', 'value')],
    Input('room-input', 'value'),
    Input('parking-input', 'value'))
def update_mapbox(budget, roominput, parkinginput):
    mapbox = px.scatter_mapbox(data[(data['price'].between(budget[0],budget[1]))&
                    (data['rooms']>=roominput)&
                    (data['parking']>=parkinginput)&
                    (data['price']/data['pred']<0.8)],
                    lat='latitude', lon='longitude',
                    color='neighbourhood', 
                    color_discrete_map={'Pinheiros': 'rgba(128,0,128,0.5)', 'Vila Madalena': 'rgba(255,0,0,0.5)', 'Butantan': 'rgba(0,128,0,0.5)'},
                    hover_data=['price', 'neighbourhood', 'link'],
                    zoom = 11,
                    mapbox_style='open-street-map')
    return mapbox


@app.callback(Output('output-pred', 'children'),
              Input('submit-button', 'n_clicks'),
              State('area-input', 'value'),
              State('region-input', 'value'))
def update_output(n_clicks, areainput, regioninput):
    if regioninput == 'Pinheiros':
        dummy = 1000
    elif regioninput == 'Vila Madalena':
        dummy = 2000
    else:
        dummy = 0
    predictedvalue = 0 + (10*areainput) + dummy if areainput != None else 0
    text = f'Your property is valued at R$ {predictedvalue:.2f}' if areainput != None else ''
    return text



if __name__ == '__main__':
    app.run_server()