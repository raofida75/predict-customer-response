# import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def explore_df(df):
    # check the shape of the dataframe
    print(f'Shape of the dataframe is: {df.shape}')
    print()
    # check if there are any missing values in the given dataframe, and return the columns which contains the missing values.
    if df.isna().any().any() == True:
        print('Missing values')
        print(df.isna().sum())
    else:
        print('There are no missing values in this dataframe.')
    print()
    # print the info of the given dataframe
    print(df.info())
    print()
    # statistical analysis of the dataframe.
    print(df.describe())
    print()
    # return the first 5 rows of the given dataframe.
    return(df.head())


def plot_hist(df, cols,bins=10):
    '''This function will return the histogram of the given numerical columns.'''
    # define subplot grid
    fig, axs = plt.subplots(nrows=1, ncols=len(cols), figsize=(15, 5))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle("Distribution in numeric columns", fontsize=18, y=0.95)
    # loop through the numeric cols and axes
    for col, ax in zip(cols, axs.ravel()):
        # get the numeric column from the dataframe and plot on specified axes
        ax.hist(df[col], bins=bins, color='lightblue')
        # chart formatting
        ax.set_title(col.upper())
        ax.grid(axis='y', alpha=0.75)
        ax.set_xlabel("")
    plt.tight_layout()
    plt.show()


def plot_bar_chart(x, y, title=''):
    color = ['lightslategray']*len(y)
    color[np.argmax(y)] = 'lightblue'
    fig = go.Figure(data=[go.Bar(x=x, y = y, marker_color=color)])
    fig.update_layout(title=title, 
                      xaxis_title='', yaxis_title='', template='plotly_white')

    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_traces(texttemplate="<b>%{x}:</b>\n%{y:.4s}")
    fig.show()


def plot_groupedBarChart(df, col, title=''):
    df=df.dropna()
    df['offer_num'] = df['offer_num'].astype('str')
    x = df[col].unique().astype('str')
    color_template = sns.color_palette("Blues", 10)
    j=0
    fig = go.Figure()
    for i in df['offer_num'].unique():
        fig.add_trace(go.Bar(
            x=x,
            y=df[df['offer_num'] == i]['success_rate'],
            name=i,
            marker_color = 'rgb'+str(color_template[j])
        ))
        j+=1
    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(title=title, xaxis_title=col, yaxis_title='Success Rate', 
                      barmode='group', xaxis_tickangle=-45, template='plotly_white',
                      height=500, width=950)
    fig.update_traces(hovertemplate='%{x}, <br>Success Rate: %{y}' )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()


def plot_bar_sub(df, cols):
    '''This function will return the bar chart of the given numerical columns.'''
    # define subplot grid
    fig, axs = plt.subplots(nrows=1, ncols=len(cols), figsize=(15, 5))
    plt.subplots_adjust(hspace=0.5)
    # loop through the numeric cols and axes
    for col, ax in zip(cols, axs.ravel()):
        vals = df[col].value_counts()
        x, y = vals.index, vals
        # get the numeric column from the dataframe and plot on specified axes
        ax.bar(x, y, color='lightblue')
        # chart formatting
        ax.set_title(col.upper())
        ax.grid(axis='y', alpha=0.75)
        ax.set_xlabel("")
    plt.tight_layout()
    plt.show()