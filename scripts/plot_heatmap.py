import os,sys
"""
# Configure the Jupyter notebook to display matplotlib plots inline.
%matplotlib inline     
"""
import matplotlib.pylab as plt
import pickle
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi']=300
# sys.path.append(os.path.expanduser("~/Projects/Github/PyComplexHeatmap"))
import PyComplexHeatmap
print(PyComplexHeatmap.__version__)
from PyComplexHeatmap import *


#set font to Arial using the following code
plt.rcParams['font.family']='sans serif'
plt.rcParams['font.sans-serif']='Arial'
# set pdf.fonttype to 42
plt.rcParams['pdf.fonttype']=42


#Generate example dataset (random)
df = pd.DataFrame(['GroupA'] * 5 + ['GroupB'] * 5, columns=['AB'])
df['CD'] = ['C'] * 3 + ['D'] * 3 + ['G'] * 4
df['EF'] = ['E'] * 6 + ['F'] * 2 + ['H'] * 2
df['F'] = np.random.normal(0, 1, 10)
df.index = ['sample' + str(i) for i in range(1, df.shape[0] + 1)]
df_box = pd.DataFrame(np.random.randn(10, 4), columns=['Gene' + str(i) for i in range(1, 5)])
df_box.index = ['sample' + str(i) for i in range(1, df_box.shape[0] + 1)]
df_bar = pd.DataFrame(np.random.uniform(0, 10, (10, 2)), columns=['TMB1', 'TMB2'])
df_bar.index = ['sample' + str(i) for i in range(1, df_box.shape[0] + 1)]
df_scatter = pd.DataFrame(np.random.uniform(0, 10, 10), columns=['Scatter'])
df_scatter.index = ['sample' + str(i) for i in range(1, df_box.shape[0] + 1)]
df_heatmap = pd.DataFrame(np.random.randn(30, 10), columns=['sample' + str(i) for i in range(1, 11)])
df_heatmap.index = ["Fea" + str(i) for i in range(1, df_heatmap.shape[0] + 1)]
df_heatmap.iloc[1, 2] = np.nan


#Annotate the rows with average > 0.3
df_rows = df_heatmap.apply(lambda x:x.name if x.sample4 > 0.5 else None,axis=1)
df_rows=df_rows.to_frame(name='Selected')
df_rows['XY']=df_rows.index.to_series().apply(lambda x:'A' if int(x.replace('Fea',''))>=15 else 'B')

row_ha = HeatmapAnnotation(
            Scatter=anno_scatterplot(df_heatmap.sample4.apply(lambda x:round(x,2)),
                            height=12,cmap='jet',legend=False,grid=True),
            Line=anno_lineplot(df_heatmap.sample4.apply(lambda x:round(x,2)),
                            height=12,colors='red',linewidth=2,legend=False),
            Bar=anno_barplot(df_heatmap.sample4.apply(lambda x:round(x,2)),
                            height=15,cmap='rainbow',legend=False),
            selected=anno_label(df_rows,colors='red',relpos=(-0.05,0.4)),
            label_kws={'rotation':30,'horizontalalignment':'left','verticalalignment':'bottom'},
            axis=0,verbose=0)

col_ha = HeatmapAnnotation(
            label=anno_label(df.AB, merge=True,rotation=10,
                             arrowprops = dict(visible=False,)
                            ), #visible in arrowprops can control whether to show the arrow
            AB=anno_simple(df.AB,add_text=True),axis=1,
            CD=anno_simple(df.CD,add_text=True),
            EF=anno_simple(df.EF,add_text=True,
                            legend_kws={'frameon':True}),
            G=anno_boxplot(df_box, cmap='jet',legend=False,grid=True),
            verbose=0)

print(np.nanmin(df_heatmap),np.nanmax(df_heatmap))

plt.figure(figsize=(5.5, 6.5))
cm = ClusterMapPlotter(
        data=df_heatmap, top_annotation=col_ha,right_annotation=row_ha,
        col_cluster=True,row_cluster=True,
        col_split=df.AB,row_split=2, z_score=0,vmin=-2.2,vmax=2.3,
        col_split_gap=0.5,row_split_gap=0.8,
        label='values',row_dendrogram=True,col_dendrogram=False,row_dendrogram_size=15,
        show_rownames=False,show_colnames=True,
        tree_kws={'row_cmap': 'Set1'},verbose=0,legend_gap=5,
        cmap='RdYlBu_r',xticklabels_kws={'labelrotation':-90,'labelcolor':'blue'})
plt.savefig("example0.pdf", bbox_inches='tight')
plt.show()
print(cm.kwargs['vmin'],cm.kwargs['vmax'],cm.legend_kws)

