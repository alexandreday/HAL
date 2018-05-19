import plotly.offline as of
from plotly.graph_objs import *
import numpy as np

def main():

    # example
    graph = {(1,2):0.5,(2,3):0.2,(3,4):0.143,(4,1):0.91}
    node_pos = {1:[0.1,0.1],2:[0.1,0.293],3:[-0.5,-0.2],4:[0.9,0.1]}
    node_score = {1:0.2993,2:0.333,3:0.999,4:0.87373}
    plot_graph(graph, node_pos, node_score)
    return

def plot_graph(graph, node_pos, node_score, fontsize=20, title='YOLO'):
    """
    Graph is just a dictionary of tuples. Values are the scores
    Node pos are the cartesian coordinate of the nodes (dict)
    """
    edge_trace_list, middle_node_trace =make_edge_trace(graph, node_pos)
    node_trace=make_node_trace(node_score, node_pos)
    node_trace2 = make_node_trace(node_score, node_pos, text="not")

    fig = Figure(data=Data([*edge_trace_list, middle_node_trace, node_trace, node_trace2]),
             layout=Layout(
                title="<br>%s"%title,
                titlefont=dict(size=fontsize),
                font = {'family' : 'CMU serif', 'size'   : fontsize},
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                #annotations=[ dict(
                    #text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                    #showarrow=False,
                    #xref="paper", yref="paper",
                    #x=0.005, y=-0.002 ) ],
                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

    of.plot(fig)

def line_interpolate(alpha, x1, x2):
    return np.array(x1)+alpha*(np.array(x2)-np.array(x1))


def make_edge_trace(graph, node_pos, lw= 2.5):
    edge_trace_list = []

    middle_node_trace = Scatter(x=[],y=[],text=[],mode='markers',hoverinfo='text', marker=Marker(opacity=0,size=2))

    for edge, v in graph.items():
        trace = Scatter(x=[],y=[],text=[],mode='lines',hoverinfo='none', line= Line(width=lw,color='#888'))
        x0, y0 = node_pos[edge[0]]
        x1, y1 = node_pos[edge[1]]
        trace['x'] += [x0, x1, None]
        trace['y'] += [y0, y1, None]
        edge_trace_list.append(trace)
        for alpha in np.linspace(0.1,0.9, 15):
            p1 = line_interpolate(alpha, node_pos[edge[0]], node_pos[edge[1]])
            middle_node_trace['x'].append(p1[0])
            middle_node_trace['y'].append(p1[1])
            middle_node_trace['text'].append("%.3f"%v)

    return edge_trace_list, middle_node_trace

def make_node_trace(node_score, node_pos, text='custom'):
    node_trace = Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    textfont = ,
    marker = Marker(
        showscale=True,
        # colorscale options
        # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
        # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
        colorscale='Electric',
        reversescale=True,
        color=[],
        size=30,
        colorbar=dict(
            thickness=10,
            title='Cluster score',
            xanchor='left',
            titleside='right'
        ),
        line=dict(width=2),
    ))

    if text =='custom':
        for k, v in node_score.items():
            x, y = node_pos[k]
            node_trace['x'].append(x)
            node_trace['y'].append(y)
            node_trace['marker']['color'].append(v)
            node_trace['text'].append("k=%i, score=%.3f"%(k,v))
    else:
        node_trace['mode'] = 'text'
        for k, v in node_score.items():
            x, y = node_pos[k]
            node_trace['x'].append(x)
            node_trace['y'].append(y)
            #node_trace['marker']['color'].append(v)
            #node_trace['marker']=Marker()#['color'].append(v)
            node_trace['text'].append(str(k))
            #node_trace['marker']['textfont']['color'].append("#000000")
        #node_trace['marker']['opacity'] = 0

    return node_trace

if __name__ == "__main__":
    main()