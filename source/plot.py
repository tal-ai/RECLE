from openTSNE import TSNE
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import row
from bokeh.layouts import gridplot
from bokeh.models import BoxSelectTool, LassoSelectTool
from bokeh.plotting import figure, curdoc
TOOLS="pan,wheel_zoom,box_select,lasso_select,reset"
output_notebook()

train_data = pd.read_csv(data_dict + 'train.csv')

# Read learned embedding
embed_data = pd.read_csv('xxx.csv')

# Dimension reduction
embedding = TSNE(perplexity=24, random_state=10).fit(embed_data.iloc[:,1:])

data_group = {
    'vote==0': (train_data.iloc[:,1] >= -1) & (train_data.iloc[:,1] <= 0),
    '1<=vote<=2': (train_data.iloc[:,1] >= 1) & (train_data.iloc[:,1] <= 2),
    '3<=vote<=4': (train_data.iloc[:,1] >= 3) & (train_data.iloc[:,1] <= 4),
    'vote==5': (train_data.iloc[:,1] >= 5) & (train_data.iloc[:,1] <= 99),
}


p_1 = figure(tools=TOOLS, plot_width=400, plot_height=400, min_border=10, min_border_left=50,
           toolbar_location="above",
           title="1 <= votes <= 4")
p_1.rect(x=embedding[data_group['1<=vote<=2'], 0], y=embedding[data_group['1<=vote<=2'] ,1], width=4, height=4, color="#AA80A0",
          width_units="screen", height_units="screen", legend=['unfluent'])

p_1.rect(x=embedding[data_group['3<=vote<=4'], 0], y=embedding[data_group['3<=vote<=4'], 1], width=4, height=4, color="#77D266",
          width_units="screen", height_units="screen", legend=['fluent'])
p_1.legend.location = "top_left"


p_0 = figure(tools=TOOLS, plot_width=400, plot_height=400, min_border=10, min_border_left=50,
           toolbar_location="above",
           title="votes = 0 or 5")

p_0.rect(x=embedding[data_group['vote==0'], 0], y=embedding[data_group['vote==0'] ,1], width=4, height=4, color="#DD5090",
          width_units="screen", height_units="screen", legend=['unfluent'])

p_0.rect(x=embedding[data_group['vote==5'], 0], y=embedding[data_group['vote==5'], 1], width=4, height=4, color="#00B2D6",
          width_units="screen", height_units="screen", legend=['fluent'])

p_0.legend.location = "top_left"


show(row(p_0, p_1))