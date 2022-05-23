"""
Script that allows an oracle to provide labels for queried pixels
"""

# Dash interface
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_table
import dash_daq as daq
import plotly.graph_objs as go
import plotly.express as px

# Basics
import numpy as np
import os
import errno
import pickle as pkl
import pdb
import sys

# Utils
from data.datasets import get_dataset
from results.utils import *
from learning.utils import *
from learning.query import QueryOutput
from path import get_path

query_file = sys.argv[1]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, update_title=None, external_stylesheets=external_stylesheets)

with open(query_file, 'rb') as f:
    history, classes, config = pkl.load(f)

dataset = get_dataset(config)

query = QueryOutput(dataset, history, classes, config)
query.patches_()

classes = build_dash_table(query.classes, transpose=True)
gt_fig = px.imshow(query.dataset.train_gt())
gt_fig.update_layout(autosize=False, width=500, height=500, coloraxis_showscale=False,
        margin=dict(l=10, r=10, b=10, t=10, pad=4), paper_bgcolor="white")

app.layout = html.Div(id='container', children=[
    html.Div(id='bandeau', children=[
        html.Div(id='classes', children=[classes]),
        html.Div(children=[dcc.Graph(id='gt', figure=gt_fig)])
    ]),
    html.Div(id='history'),
    html.Div(id='iteration'),
    dcc.Store(id='iteration_state', storage_type='memory', data=0),
    html.Div(id='px-id'),
    html.Div(className='vertical-box', children=[
        html.Div(className='box-1', children=[
            dcc.Graph(id='RGB'),
        ], style={'display': 'flex', 'justify-content': 'space-around'}),
        html.Div(className='box-1', children=[
            #dcc.Graph(id='clusters'),
            dcc.Graph(id='spectrum')
        ], style={'display': 'flex', 'justify-content': 'space-around'})
    ]),
    html.Div(className='horizontal-box', children=[
        html.Div(id='annotation-button', children=[
            dcc.Dropdown(
                id='class',
                options=[{'label': label, 'value': i} for i, label in enumerate(dataset.label_values)],
                value=0,
                style={'width': '20em'})
        ]),
        html.Button('Label', id='class-btn', n_clicks=0),
        html.Div(id='add-new-class', children=[
            dcc.Input(id='new-class'),
            html.Button('Add new class', id='new', n_clicks=0)
        ])
    ]),
    dcc.Checklist(
        id='class-checkbox',
        options=[{'label': label_, 'value': class_id} for class_id, label_ in enumerate(dataset.label_values[1:])],
        value=[0],
        labelStyle={'display': 'inline-block'}),
    dcc.Graph(id='mean_spectra'),
    dcc.Store(id='annotation', storage_type='memory'),
    html.Div(id='annotation-div'),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,
        n_intervals=0
    )], style={'margin': '50px 50px'})

@app.callback(Output('mean_spectra', 'figure'),
                Input('class-checkbox', 'value'))
def update_mean(classes):
    mean_sp_graph = go.Figure()
    mean_spectra = query.dataset.mean_spectra_
    for class_id in classes:
        sp = mean_spectra[class_id+1]
        sp = sp.reshape(1, sp.shape[0])
        sp = spectra_bbm(sp, query.dataset.sensor.mask_bands)
        sp = sp.reshape(-1)
        mean_sp_graph.add_trace(go.Scatter(y=sp, mode='lines'))#, name='{}'.format(query.dataset.label_values[class_id])))
    return mean_sp_graph

@app.callback([Output('RGB', 'figure'), Output('spectrum', 'figure'), Output('px-id', 'children')],
                Input('class-btn', 'n_clicks'))
def update_rgb(n):
    patch_id = min(query.patch_id, len(query.patches.items())-1)
    patch = query.patches[patch_id]
    patch_x, patch_y = query.patch_coordinates['x'][patch_id], query.patch_coordinates['y'][patch_id]
    x, y = query.coordinates[patch_id]
    query.x = x
    query.y = y
    rgb = hyper2rgb(patch, query.dataset.rgb_bands)

    rgb[patch_x, patch_y,:] = [255, 0, 0]
    fig = px.imshow(rgb)
    # cluster = px.imshow(query.regions[patch_id])

    spectrum = query.dataset.IMG[x,y,:]
    spectrum = spectrum.reshape(1, spectrum.shape[0])
    spectrum = spectra_bbm(spectrum, query.dataset.sensor.mask_bands)
    spectrum = spectrum.reshape(-1)

    sp = go.Figure(data=go.Scatter(y=spectrum))
    wavelengths = query.dataset.sensor.wavelengths
    if wavelengths is None:
        wavelengths = np.arange(1, dataset.n_bands+1) 
    ticks = np.linspace(0,len(wavelengths)-1,20)

    sp.update_xaxes(tickangle=45,
                    title_text = 'Wavelength (µm)',
                    title_font=dict(size=20, color='black'),
                    tickmode = 'array',
                    tickvals = ticks,
                    ticktext= [wavelengths[int(i)] for i in ticks],
                    showline=True,
                    linecolor='#808080',
                    tickfont=dict(size=18, color='black'), showgrid=True, gridcolor='#808080',)

    sp.update_yaxes(title_text = "Reflectance", title_font=dict(size=20, color='black'), \
                    tickfont=dict(size=18, color='black'), showgrid=True, gridcolor='#808080', \
                    showline=True, linecolor='#808080')

    fig.update_layout(autosize=False, width=500, height=500,
            margin=dict(l=10, r=10, b=10, t=10, pad=4), paper_bgcolor="white")


    # cluster.update_layout(autosize=False, width=500, height=500, coloraxis_showscale=False,
    #         margin=dict(l=10, r=10, b=10, t=10, pad=4), paper_bgcolor="white")

    query.patch_id += 1

    if query.patch_id > len(query.patches.items()):
        nb_px = html.P('Retrain model')
    else:
        nb_px = html.P('Pixel {}/{}'.format(query.patch_id, len(query.patches.items())))

    return fig, sp, nb_px

@app.callback(Output('annotation', 'data'),
                Input('class', 'value'))
def update_gt(label_id):
    if label_id == None:
        raise PreventUpdate
    query.annotation = int(label_id)
    return label_id

@app.callback([Output('annotation-div', 'children'), Output('classes', 'children')],
                Input('class-btn', 'n_clicks'))
def update_gt(n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'class-btn.n_clicks' in changed_id and query.n_added < query.n_px:
        query.label()
        classes = build_dash_table(query.classes, transpose=True)
        query.save()
        return None, classes
    else:
        raise PreventUpdate

@app.callback(Output('annotation-button', 'children'),
                [Input('new-class', 'value'), Input('new', 'n_clicks')])
def add_class(new_class, n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'new.n_clicks' in changed_id:
        query.add_class(new_class)
        
        annotation_children = [
            dcc.Dropdown(
                id='class',
                options=[{'label': label, 'value': i} for i, label in enumerate(query.dataset.label_values)],
                value=0,
                style={'width': '20em'})
            ]

        return annotation_children
    else:
        raise PreventUpdate


if __name__ == '__main__':
    app.run_server(debug=True)
