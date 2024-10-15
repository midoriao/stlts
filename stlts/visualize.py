import numpy as np
import pandas as pd
import plotly.express as px

from stlts.trace import Trace


def plot_rss(trace_df: pd.DataFrame, interpolate: bool = False):
    from stlts.rss_scenario import car_length, car_width

    if interpolate:
        trace = Trace.from_df(trace_df).interpolate().df
    else:
        trace = trace_df.copy()

    trace['distance_x'] = abs(trace['bx'] - trace['ax'])
    trace['distance_y'] = abs(trace['by'] - trace['ay'])
    trace['distance'] = np.maximum(
        abs(trace['by'] - trace['ay']) - car_width,
        abs(trace['bx'] - trace['ax']) - car_length,
    )

    trace['danger'] = -np.maximum.reduce([
        trace['bx'] - trace['ax'] - trace['rssDistance_a'] - car_length,
        trace['ax'] - trace['bx'] - trace['rssDistance_b'] - car_length,
        trace['by'] - trace['ay'] - trace['lateralRssDistance_a'] - car_width,
        trace['ay'] - trace['by'] - trace['lateralRssDistance_b'] - car_width,
    ])
    trace_a = pd.DataFrame(
        dict(
            t=trace['t'],
            x=trace['ax'],
            y=trace['ay'],
            x_vel=trace['ax_vel'],
            x_acc=trace['ax_acc'],
            y_vel=trace['ay_vel'],
            y_acc=trace['ay_acc'],
            car='a',
        )
    )

    trace_b = pd.DataFrame(
        dict(
            t=trace['t'],
            x=trace['bx'],
            y=trace['by'],
            x_vel=trace['bx_vel'],
            x_acc=trace['bx_acc'],
            y_vel=trace['by_vel'],
            y_acc=trace['by_acc'],
            car='b',
        )
    )

    px.line(
        trace,
        x='t',
        y=[
            'distance',
            'rssDistance_a',
            'rssDistance_b',
            'lateralRssDistance_a',
            'lateralRssDistance_b',
            'danger',
        ],
        markers=not interpolate,
    ).show()

    px.line(
        trace,
        x='t',
        y=[
            'ax',
            'ay',
            'ax_vel',
            'ax_acc',
            'ay_vel',
            'ay_acc',
            'bx',
            'by',
            'bx_vel',
            'bx_acc',
            'by_vel',
            'by_acc',
        ],
        markers=not interpolate,
    ).show()

    fig = px.line(
        pd.concat(
            [trace_a, trace_b],
            ignore_index=True,
        ),
        x='x',
        y='y',
        color='car',
        hover_data=['t'],
        markers=not interpolate,
    )

    for time in [0, trace['danger'].argmax(), len(trace) - 1]:
        fig.add_shape(
            type="rect",
            x0=trace['ax'][time],
            y0=trace['ay'][time],
            x1=trace['ax'][time] + car_length,
            y1=trace['ay'][time] - car_width,
            fillcolor="LightSkyBlue",
            line=dict(color="RoyalBlue"),
        )
        fig.add_shape(
            type="rect",
            x0=trace['bx'][time],
            y0=trace['by'][time],
            x1=trace['bx'][time] + car_length,
            y1=trace['by'][time] - car_width,
            fillcolor="LightSalmon",
            line=dict(color="Tomato"),
        )
    for l in [0, 3.5, 7, 10.5]:
        fig.add_hline(y=l, line_width=1, line_dash="dash", line_color="gray")
    fig.show()
