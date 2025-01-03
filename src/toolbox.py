import plotly.graph_objects as go
import numpy as np


def plot_scatters(exp_name, data, log_x=False):
    fig = go.Figure()

    for i, (name, values) in enumerate(data.items()):
        fig.add_trace(
            go.Scatter(
                x=values,
                y=np.full_like(values, i),
                mode="markers",
                name=name,
            )
        )

    fig.update_layout(showlegend=True)
    if log_x:
        fig.update_xaxes(type="log")
    fig.write_html(f"plots/{exp_name}.html")
    return fig


def str_of_eval(domain_randomization, use_slow_env, deterministic):
    domain = "rnd" if domain_randomization else "det"
    env = " slow" if use_slow_env else ""
    sampling = "det" if deterministic else "stochastic"
    return f"{domain}{env} env with {sampling} policy"
