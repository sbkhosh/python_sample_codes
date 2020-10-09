#!/usr/bin/python3

def calc_in_out_lr(x, x_in, x_out):
    
    p0s_in = []; p1s_in = []
    p0s_out = []; p1s_out = []

    for i in range(0, len(x.index)):
        
        asset_i = x.iloc[i]
        
        y_in = asset_i.values[0:11]    
        y_out = asset_i.values[12:24]
    
        lm_in = regression.linear_model.OLS(y_in, x_in).fit()   
        p0s_in.append(lm_in.params[0])
        p1s_in.append(lm_in.params[1])    
    
        lm_out = regression.linear_model.OLS(y_out, x_out).fit()
        p0s_out.append(lm_out.params[0])
        p1s_out.append(lm_out.params[1])  
        
    df = pd.DataFrame(list(zip(p0s_in, p1s_in, p0s_out, p1s_out)), index=x.index, 
                      columns=['p0_in', 'p1_in', 'p0_out', 'p1_out'])   

    return(df)


    def plot_hist2d(ax, p0_name, p1_name, title):
    x = winsorize(df_lr[p0_name].values, limits=[0.01, 0.01])
    y = winsorize(df_lr[p1_name].values, limits=[0.01, 0.01])
    
    x_bins = np.linspace(np.min(x),np.max(x), 40)
    y_bins = np.linspace(np.min(y),np.max(y), 40)
    
    ax.hist2d(df_lr[p0_name], df_lr[p1_name], bins=[x_bins, y_bins], cmap=cm.Blues)
    ax.set_xlabel(p0_name)
    ax.set_ylabel(p1_name)
    ax.set_title(title)
    
fig, axs = plt.subplots(1, 2, sharey=False, figsize=(14, 6))
plot_hist2d(axs[0], 'p0_in', 'p1_in', 'In-Sample')
plot_hist2d(axs[1], 'p0_out', 'p1_out', 'Out-of-Sample')

n_clusters = 4

km_data_in = np.matrix([df_lr['p0_in'].values, df_lr['p1_in'].values])

km_in = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300)
km_in.fit(km_data_in.T)

y_km_in = km_in.predict(km_data_in.T)


idx = np.argsort(km_in.cluster_centers_.sum(axis=1))
lut = np.zeros_like(idx)
lut[idx] = np.arange(4)

def set_cluster(x, cluster_labels):
    global indx
    df = pd.DataFrame(index=x.index)   
    df['cluster'] = cluster_labels[indx : indx + len(x.index)]   
    indx += len(x.index)
    return df

indx = 0
grouper = [mom10_rtns_bull.index.get_level_values('date')]
df_clusters = mom10_rtns_bull.groupby(grouper).apply(lambda x: set_cluster(x, lut[y_km_in]))
