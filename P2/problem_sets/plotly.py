import plotly.plotly as py
import plotly.graph_objs as go

def plot_heatmap():
    data = [
        go.Heatmap(
            z=[[1, 20, 30],
               [20, 1, 60],
               [30, 60, 1]]
        )
    ]
    plot_url = py.plot(data, filename='basic-heatmap')

plot_heatmap()
