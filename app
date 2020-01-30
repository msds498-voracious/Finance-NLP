# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 05:48:29 2020

@author: glawson
"""


"""
*******************************************************************************
To view the dashboard:
    1) Save the script in a location that you can easily navigate to using the Python 
        terminal.  I used my C: Drive because I couldn't figure out how to get
        to my Google drive.
    2) In your Python terminal, use cd to get to the appropriate directory
    3) Input "python app.py" once you are in the correct directory where the file
        lives.
    4) Visit http://127.0.0.1:8050/ in your web browser
*******************************************************************************
"""

import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Voracious'),

    html.Div(children='''
        A dashboard for creating investment insight from social media.
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'Apple'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Facebook'},
            ],
            'layout': {
                'title': 'Stock Sentiment Graphic'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
    

