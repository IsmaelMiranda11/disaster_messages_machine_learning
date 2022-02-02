import pandas as pd
from sqlalchemy import create_engine
import plotly.express as pex
from plotly import graph_objects as go

# Connecting to dataset in sqlite
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# Creating the graphs

def create_graphs(df):
    '''
    Function to create graphs with the dataset data
    Input
        None
    Output
        figures - list of graph object of plotly
    '''
    
    figures = []

    # ------- First Graph - Number of messages by genre -------
    # Data
    genre_df = pd.DataFrame(df.genre.value_counts().reset_index())
    genre_df.columns = ['genre', 'count']

    # Create the graph with plotly express
    g = pex.bar(genre_df.sort_values('count'),  
        y='genre', 
        x='count', 
        orientation='h')

    # Adjusting layout
    g.update_layout(title='Number of messages by genre',
        title_font_size=20)#, font_size=16)
    g.update_traces(textfont_size=20, textposition='auto', 
        textangle=0, insidetextanchor='middle', texttemplate='# %{x:,.0f}',
        marker_color='#459b45')
    g.update_yaxes(title='', tickfont_size=20)

    figures.append(g)

    # ------- Second Graph - Percentage of messages by category -------
    # Data
    cate_df = pd.DataFrame(df.iloc[:,-36:].mean(), columns=['valor'])
    cate_df = cate_df.sort_values('valor').round(2)
    # Create the graph with plotly express

    # Data labels
    texts = []
    for k, v in dict(zip(cate_df.index, cate_df.valor)).items():
        texts.append(str(k) + ': ' + str(round((v*100),2)) + '%')

    # Graph
    g = pex.bar(cate_df, x=cate_df.valor, y=cate_df.index, text=texts, 
        title='Percentage of messages by category'
        )
    
    # Adjusting layout
    g.update_layout(plot_bgcolor='white', barmode='stack', bargap=0.1,
        title_font_size=20)
    g.update_yaxes(showticklabels=False, title='', side='right')
    g.update_xaxes(range=[1,0], showticklabels=False, title='')
    g.update_traces(textposition='outside', textfont_size=20,
        marker_color='#459b45')

    figures.append(g)

    # ------- Thrid Graph - Indicators with whole number -------
    # Data
    qtd_msg = df.shape[0]
    qtd_cat = 36

    # Using graph objects of plotly
    ind_1 = go.Indicator(
        mode='number',
        value=qtd_msg,
        number={'valueformat':',.0f', 'font_size':55},
        domain={'row': 0, 'column': 0},
        title={'text':'Number of messages analyzed', 'align': 'left', 
            'font_size':24}
    )

    ind_2 = go.Indicator(
        mode='number',
        value=qtd_cat,
        number={'valueformat':'.0f', 'font_size':55},
        domain={'row': 1, 'column': 0},
        title={'text':'Number of messages categories', 'align': 'left', 
            'font_size':24}
    )
    
    # Graph
    g = go.Figure()
    g.add_traces(ind_1)
    g.add_traces(ind_2)

    # Adjusting layout
    g.update_layout(width=500, grid={'rows':2, 'columns':1}, separators='.,' )

    figures.append(g)

    return figures
    