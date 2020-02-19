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
from dash.dependencies import Input, Output
import base64
import numpy as np
import pandas as pd
from yahoo_historical import Fetcher

from get_tweet_data import getrawtweets

from dateutil.relativedelta import relativedelta                                                                                                                  
from datetime import date 

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Create the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    #'text': '#347235',
    'sidebar': '#565051',
    'text': '#1589FF'
}

#Add logo image to page
image_filename = 'voracious2.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read()).decode('ascii')


five_yrs = relativedelta(months=60) 
today = date.today()#
five_yrs_ago = today - five_yrs
five_yrs_ago=five_yrs_ago.strftime('%Y-%m-%d')
yr=int(five_yrs_ago[0:4])
mo=int(five_yrs_ago[5:7])
dy=int(five_yrs_ago[8:10])

#******************************************************************************
#Create the Layout
#******************************************************************************

app.layout = html.Div( children = [ 
    #Logo
    html.Div( children=[
        html.Img(src='data:image/png;base64,{}'.format(encoded_image))
        ], style={'width': '10%',
                    'float': 'right',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    
                    #'color':colors['text']
                    #'backgroundColor': colors['sidebar']
                    }),
    
    #Title Block Header    
    html.Div( children=[
        html.H1('Investing through AI and Social Media'),
        #html.P(''),
        #html.P('')
        ], style={'width': '90%',
                    'display': 'inline-block',
                    'color':colors['text']
                    #'backgroundColor': colors['sidebar']
                    }),
    
    #Dropdown and prediction table
    html.Div( children=[
        # First let users choose stocks
        html.H2('Choose a stock ticker'),
        dcc.Dropdown(
            id='my-dropdown',
            options=[{'label': 'Apple', 'value': 'AAPL'},
                     {'label': 'Amazon', 'value': 'AMZN'},
                     {'label': 'Facebook', 'value': 'FB'},
                     {'label': 'Google', 'value': 'GOOGL'},
                     {'label': 'Netflix', 'value': 'NFLX'}
                     ],
            value='AAPL'),
        html.P(''),
        #html.H2('Future Price Movement Predictions'),
        html.H2(id='future-prediction-title'),
        html.Table(id='my-table'),
        #html.Table(id='my-table'),
        html.P(''),
        ], style={'width': '35%',
                    'display': 'inline-block'#,
                    #'backgroundColor': colors['sidebar']
                    }),
        
    #Graph the stock price in the top right corner
    html.Div( children = [
        #html.H2('5 years stocks price graph'),
        html.H2(id='stock-graph-title'),
        dcc.Graph(
                figure={'layout':{
                            'title':'Stock Price over Time',
                            'xaxis':{
                                'title':'Time'
                                },
                            'yaxis':{
                                'title':'Daily Close Price (USD)'
                                }
                        }}, id='my-graph'),
        html.P('')
        ], style={'width': '50%',
                    'float': 'right',
                    'display': 'inline-block'
                    #'backgroundColor': colors['sidebar']
                    }),

    #Tweet table
    html.Div( children = [
        html.P(''),
        #html.H4('Recent News'),
        html.H2(id='recent-news-title'),
        html.Table(id='stock-tweets-table')
        ], style={'width': '100%', 
                    'float': 'right', 
                    'display': 'inline-block'}),
    ]
)


#******************************************************************************
#Define the Callbacks
#******************************************************************************

#Stock Graph Title
@app.callback(
    Output('stock-graph-title', 'children'),
    [Input('my-dropdown', 'value')]
)
def update_graph_title(x):
    return "5-Year Stock Data Graph for {}".format(x)

#Recent News Title
@app.callback(
    Output('future-prediction-title', 'children'),
    [Input('my-dropdown', 'value')]
)
def update_fut_pred_title(x):
    return "Future {} Price Movement Predictions".format(x)

#Recent News Title
@app.callback(
    Output('recent-news-title', 'children'),
    [Input('my-dropdown', 'value')]
)
def update_recent_news_title(x):
    return "Recent {} News".format(x)


#Callback for the stocks graph
@app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    global stockpricedf # Needed to modify global copy of stockpricedf
    ticker_fb = Fetcher(selected_dropdown_value, [yr,mo,dy])
    #ticker_fb = Fetcher('fb', [yr,mo,dy])
    stockpricedf=ticker_fb.get_historical()
    stockpricedf['Date'] = pd.to_datetime(stockpricedf['Date'])
    return {
        'data': [{
            'x': stockpricedf.Date,
            'y': stockpricedf.Close
        }]
    }

#Callback for the tweet table
@app.callback(Output('stock-tweets-table', 'children'), [Input('my-dropdown', 'value')])
def generate_table(selected_dropdown_value,max_rows=10):
    global tweetdf # Needed to modify global copy of financialreportingdf
    tweetdf = getrawtweets(selected_dropdown_value)
    tweetdf['Sentiment'] = np.round(tweetdf['Sentiment'],2)
    
    # Header
    return html.Table([html.Tr([html.Th(col) for col in tweetdf.columns])] + [html.Tr([
        html.Td(tweetdf.iloc[i][col]) for col in tweetdf.columns
    ]) for i in range(min(len(tweetdf), max_rows))])


#******************************************************************************
#Run the App
#******************************************************************************

if __name__ == '__main__':
    app.run_server(debug=True)
   
