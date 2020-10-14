#!/usr/bin/python3

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table

from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP],meta_tags=[{"content": "width=device-width"}])
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    html.Div(id='target'),
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'Link 1', 'value': 'link1'},
            {'label': 'Link 2', 'value': 'link2'},
        ],
        value='link1'
    )
])


@app.callback(Output('target', 'children'), [Input('dropdown', 'value')])
def embed_iframe(value):
    links = {
        'link1': 'long-term-trends',
        'link2': 'price-surprises',
    }
    return html.Iframe(src=f'https://www.barchart.com/futures/{links[value]}')


if __name__ == '__main__':
    app.run_server(debug=True)
