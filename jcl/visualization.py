import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from_list = LinearSegmentedColormap.from_list
from matplotlib.cm import ScalarMappable
from jcl.utils import trial_distance, trial_duration
from jcl.analysis import Map
import plotly.express as px
import plotly.graph_objects as go

def plot_map(m: Map, title=None, colorbar_label=None, path=None):
    """ Plot given with added color bar."

        Args:
            m - map
            title - figure title
            colorbar_label - label to be used for the colorbar
            path - file path to which to save the figure, if none show the figure
    """
    if m.ndim == 1:
        im = np.expand_dims(m.map, 1)
        im = np.repeat(im.map, 3, 1).T
    else:
        im = m.map

    plt.set_cmap("jet")
    plt.imshow(im)

    cb = plt.colorbar(ticks=[0., np.max(im)])
    if colorbar_label is not None:
        cb.set_ticklabels([0, np.round(np.max(im))])
        cb.set_label("Hz", fontsize=18)

    if title is not None:
        plt.title(title)

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()
    
def plot_map_with_label(m: Map, start=(a,b), chocolate = (c,d), honey=(e,f), title=None, colorbar_label=None, path=None):
    """ Plot given with added color bar and marking on start, chocolate and honey"

        Args:
            m - map
            title - figure title
            colorbar_label - label to be used for the colorbar
            path - file path to which to save the figure, if none show the figure
    """
    if m.ndim == 1:
        im = np.expand_dims(m.map, 1)
        im = np.repeat(im.map, 3, 1).T
    else:
        im = m.map
        
    # Create a figure and axis
    fig, ax = plt.subplots()

    plt.set_cmap("jet")
    plt.imshow(im)

    # Add the label text in black
    ax.text(a, b, "Start", fontsize=12, color='white', ha='center', va='center')
    ax.text(c, d, "Chocolate", fontsize=12, color='white', ha='center', va='center')
    ax.text(e, f, "Honey", fontsize=12, color='white', ha='center', va='center')

    cb = plt.colorbar(ticks=[0., np.max(im)])
    if colorbar_label is not None:
        cb.set_ticklabels([0, np.round(np.max(im))])
        cb.set_label("Hz", fontsize=18)

    if title is not None:
        plt.title(title)

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()    

def plot_circle(center, radius, color="red", name=None, fig=None):
    """ Plot circle.

        Args:
            circle - ((c_x, c_y), r1, color=None)
            center - (c_x, c_y)
            redius - radius
            color - line color
            name - name to show in the legend, default is None
            fig - figure to plot on, if None (default) create new
        Return:
            fig
    """
    if fig is None:
        fig = go.Figure()

    if radius > 0:
        x0, x1 = center[0] - radius, center[0] + radius
        y0, y1 = center[1] - radius, center[1] + radius
        fig.add_shape(type="circle", xref='x', yref='y',
                      x0=x0, x1=x1, y0=y0, y1=y1,
                      line_color=color)
    scat_args = dict(x=[center[0]], y=[center[1]],
                     marker_color=color, marker_size=15, showlegend=False)
    if name is not None:
        scat_args["name"] = name
        scat_args["showlegend"] = True

    fig.add_trace(go.Scatter(**scat_args))
    return fig


def plot_trajectory(trajectory, duration=None, title=None, path=None, ax=None, trajectory_2=None, start_box=None, speed=None):
    """ Plot given trajectory.

        Args:
            trajectory - animal trajectory during a trial, array of shape (n,2)
            duration - trial duration in seconds
            title - figure title
            path - file path at which to save the figure; if `None` and ax is `None` show the figure
            ax - axes object for plotting (plotly figure); if `None` creates a new one
            trajectory_2 - second animal trajectory during a trial, array of the same shape as trajectory
            start_box - coordinates of the start box [(TL_x, TL_y), (BR_x, BR_y)]
            speed - speed of the animal, can be passed only if `duration` is None
        Return:
            Figure with plotted trajectory.
    """

    if duration is not None and speed is not None:
        raise ValueError("Pass only `duration` OR `speed`.")

    if speed is None:
        color = np.arange(len(trajectory)) / len(trajectory)
        if duration is not None:
            color = color * duration
    else:
        color = speed

    fig = px.scatter(x=trajectory[:, 0], y=trajectory[:, 1],
                     color=color)

    if ax is not None:
        t = list(fig.select_traces())[0]
        ax.add_trace(t)
        fig = ax

    if speed is None and duration is None:
        fig.update_layout(coloraxis_showscale=False)
    elif speed is not None:
        fig.update_layout(coloraxis_colorbar_title_text="Speed (cm / s)")
    else:
        fig.update_layout(coloraxis_colorbar_title_text="Time (s)")

    if ax is None:
        fig.update_xaxes(showgrid=False, visible=False)
        fig.update_yaxes(showgrid=False, visible=False)
        fig.update_layout(template="plotly_white")
        # if figure is provided we don't want to change its style

    if trajectory_2 is not None:
        t2t = go.Scatter(x=trajectory_2[:, 0], y=trajectory_2[:, 1],
                         mode="lines", showlegend=False, line_color="green")
        fig.add_trace(t2t)

    if start_box is not None:
        (TL_x, TL_y), (BR_x, BR_y) = start_box
        sc_x = [TL_x, BR_x, BR_x, TL_x, TL_x]
        sc_y = [BR_y, BR_y, TL_y, TL_y, BR_y]
        fig.add_trace(go.Scatter(x=sc_x, y=sc_y,
                                 line={"color": "black"}, showlegend=False))

    if title is not None:
        fig.update_layout(title=title)

    if path is not None:
        fig.write_image(path)
    return fig


def plot_trials_in_session(trial_inds, sampling_rate=None, path=None):
    """ Visualize trials in within recording session.

        Args:
            trial_inds - list of trial indices (list of tuples)
            sampling_rate - whl sampling rate, if None (default) time won't be converted to seconds
            path - path to which save the image
    """
    if sampling_rate is not None:
        trial_inds = [(t[0] / sampling_rate, t[1] / sampling_rate) for t in trial_inds]
        plt.xlabel("Time (s)")

    for i, t in enumerate(trial_inds):
        plt.fill_betweenx([0, 1], t[0], t[1], color="orange", alpha=.45)
        txt = f"Trial {i+1}"
        if sampling_rate is not None:
            txt += f" ({np.round(t[1]-t[0], 2)}s)"
        plt.text((t[0]+t[1])/2, .5, txt, rotation="vertical")

    plt.yticks([])

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


"""
# quick test
# TODO unit tests

plot_place_field(np.array([0,0,0,1,2,3,2,1,0,0,0,0,0,0]))
plot_place_field(np.array([[0,0,0,1,2,3,2,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,2,3,2,1,0,0,0]]))
"""
