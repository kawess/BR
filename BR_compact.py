import matplotlib
#matplotlib.use('TkAgg')  # If interactive
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import pandas as pd
import io
import seaborn as sn
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import numpy as np
import json

A4_landscape = (11.69, 8.27)

config_file = open('config.json')
tables = json.loads(config_file.read())
config_file.close()

shapefile = tables['input_files']['shapefile']
datafile = tables['input_files']['data']
depth_intervals = tables['input_files']['depth_intervals']
output_folder = tables['output_folder']

d_intervals = pd.read_csv(depth_intervals, delimiter='\t')

# Open the file and read lines skipping rows starting with '//' these are comments
with open(datafile, 'r', encoding='utf-8') as file:
     lines = [line for line in file if not line.startswith('//')]

# Convert the filtered lines into a pandas DataFrame
df = pd.read_csv(io.StringIO(''.join(lines)), delimiter='\t', dtype={'Cruise': str,
                                                                     'Station': str,
                                                                     'Type': str})
# Rename date column
df.rename(columns={"yyyy-mm-ddThh:mm:ss.sss": "datetime"}, inplace=True)
df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')

# Give the columns with QF dtype string (before QF could be both, and not match was not correct)
cols = df.columns[df.columns.str.startswith('QV:SEADATANET')]
df[cols] = df[cols].astype(str)

#####################################################################
# Shapefile
#####################################################################

gdf = gpd.read_file(shapefile)
# Convert map to WGS84
gdf_wgs84 = gdf.to_crs('EPSG:4326')

# Specific for attributes in the shape
gdf_wgs84['Open_sea'] = gdf_wgs84['level_2']

#gdf_wgs84.loc[gdf_wgs84['HELCOM_ID'] == 'SEA-013A', 'Open_sea'] = 'Gulf of Finland A'
#gdf_wgs84.loc[gdf_wgs84['HELCOM_ID'] == 'SEA-013B', 'Open_sea'] = 'Gulf of Finland B'

areas = gdf_wgs84['Open_sea'].dropna().unique()  # Basin names

#fig, ax = plt.subplots(figsize=(10, 8))
#gdf_wgs84[gdf_wgs84["Open_sea"].notna()].plot(ax=ax)
# #gdf_wgs84.plot(ax=ax)
#plt.show()

fig_map, ax_map = plt.subplots(figsize=(8, 8))

gdf_wgs84.plot(
    ax=ax_map,
    color='lightgray'
)

ax_map.set_axis_off()

fig_map.canvas.draw()

map_image = np.asarray(fig_map.canvas.buffer_rgba())
map_extent = (
    *ax_map.get_xlim(),
    *ax_map.get_ylim()
)

#plt.show()
#plt.close(fig_map)

############################################################################
# Match position with area
############################################################################
print('Matching coordinates')

# Only rows that actually contain a profile position
df_positions = df.dropna(
    subset=[
        'CDI-record id',
        'Longitude [degrees_east]',
        'Latitude [degrees_north]'
    ]
).copy()

# Create points only for profile positions
points = gpd.GeoDataFrame(
    df_positions,
    geometry=gpd.points_from_xy(
        df_positions['Longitude [degrees_east]'],
        df_positions['Latitude [degrees_north]']
    ),
    crs='EPSG:4326'
)

# Spatial join only the profile positions
profiles_joined = gpd.sjoin(
    points,
    gdf_wgs84[['Open_sea', 'geometry']].dropna(subset=['Open_sea']),
    predicate='intersects',     # eller exkludera: 'within',
    how='left'
)

# Keep one match per profile
profile_info = profiles_joined[
    ['CDI-record id', 'Open_sea', 'geometry']
].drop_duplicates('CDI-record id')

# Start with ALL original rows
df_joined = df.copy()

# Fill profile information down
df_joined[
    ['CDI-record id',
     'Longitude [degrees_east]',
     'Latitude [degrees_north]',
     'datetime']
] = (
    df_joined[
        ['CDI-record id',
         'Longitude [degrees_east]',
         'Latitude [degrees_north]',
         'datetime']
    ].ffill()
)

# Add the matched sea area
df_joined['Open_sea'] = df_joined['CDI-record id'].map(
    profile_info.set_index('CDI-record id')['Open_sea']
)

# Add the matched geometry
df_joined['geometry'] = df_joined['CDI-record id'].map(
    profile_info.set_index('CDI-record id')['geometry']
)

# Make sure it remains a GeoDataFrame
df_joined = gpd.GeoDataFrame(
    df_joined,
    geometry='geometry',
    crs='EPSG:4326'
)

############################################################################
############################################################################

parameters =  list(df.columns[9::2])[:-1] #    Or chose 1! # ['Water body nitrate [umol/l]'] #['Water body phosphate [umol/l]'] #
br_min_notzero = ['Water body salinity [per mille]',
                  'ITS-90 water temperature [degrees C]',
                  'Water body total alkalinity [mEquiv/l]',
                  'Water body pH [pH units]',
                  'Water body total nitrogen [umol/l]',
                  'Water body total phosphate [umol/l]',
                  'Water body dissolved oxygen concentration [umol/l]',
                  'Water body uncalibrated oxygen sensor data [umol/l]',
                  'Water body dissolved oxygen saturation [%]']

for parameter in parameters:
    print(parameter)
    col = df_joined.columns.get_loc(parameter)
    stations =  df_joined.loc[df_joined.iloc[:, col + 1].isin(['1', '2', '5', '6', '8', 'Q'])].copy()

    # Where to save output
    save_path = os.path.join(output_folder, parameter.partition("[")[0].strip())
    if not os.path.isdir(save_path):
         os.makedirs(save_path)

    pdf = PdfPages(os.path.join(save_path, f"{parameter.split(' [')[0]}.pdf"))

    # Define depth interval
    print('Fixing depth intervals')
    ## Standard
    bins = [0, 10, 20, 30, 50, 100, 150, 200, 250]
    labels = ['0–10', '10–20', '20–30', '30–50', '50–100', '100–150', '150–200', '200–250']

    stations['Depth_int_standard'] = pd.cut(
        stations['Depth [m]'],
        bins=bins,
        labels=labels,
        right=False
    )

    # Area specific
    stations = stations.merge(
        d_intervals[['Open_sea', 'sur_min', 'sur_max', 'deep_min', 'deep_max']],
        on='Open_sea',
        how='left'
    )

    depth = stations['Depth [m]']

    stations['Depth_int'] = pd.NA

    # Surface interval
    mask = (
            depth.notna()
            & stations['sur_min'].notna()
            & (depth >= stations['sur_min'])
            & (depth <= stations['sur_max'])
    )

    stations.loc[mask, 'Depth_int'] = (
            stations.loc[mask, 'sur_min'].astype(int).astype(str)
            + '–'
            + stations.loc[mask, 'sur_max'].astype(int).astype(str)
    )

    # Deep interval
    mask = (
            depth.notna()
            & stations['deep_min'].notna()
            & (depth > stations['deep_min'])
    )

    # Deep → bottom
    mask_bottom = (
            mask
            & stations['deep_max'].astype(str).str.lower().eq('bottom')
    )

    stations.loc[mask_bottom, 'Depth_int'] = (
            stations.loc[mask_bottom, 'deep_min'].astype(int).astype(str)
            + '–bottom'
    )

    # Deep → numerical maximum
    mask_deep = mask & ~mask_bottom

    stations.loc[mask_deep, 'Depth_int'] = (
            stations.loc[mask_deep, 'deep_min'].astype(int).astype(str)
            + '–'
            + stations.loc[mask_deep, 'deep_max'].astype(int).astype(str)
    )

    stations['Month'] = stations['datetime'].dt.month

    # Statistics per area and depth interval
    print('Calculate statistics')
    df_stats = (
        stations
        .dropna(subset=[parameter])
        .groupby(['Open_sea', 'Depth_int'], observed=True)[parameter]
        .agg(
            n='count',
            min='min',
            max='max',
            std='std',
            minstd=lambda x: x.min() - x.std() if parameter in br_min_notzero else 0,
            maxstd=lambda x: round(x.max() + x.std(), 2),
            min10=lambda x: round(x.min() * 0.9, 2) if parameter in br_min_notzero else 0,
            max10=lambda x: round(x.max() * 1.1, 2),
            max_id=lambda x: int(stations.loc[x.idxmax(), 'CDI-record id'])
        )
        .reset_index()
    )

    df_stats.to_csv(
        os.path.join(save_path, f'{parameter.partition("[")[0].strip()}_stats.csv'),
        index=False,
        encoding='utf-8-sig', sep=';',  float_format='%.2f',
    )

    # Depth variation - all data
    fig, ax = plt.subplots(1,2, figsize=A4_landscape)
    sn.boxplot(ax=ax[0],
        data=stations,
        x='Depth_int_standard',
        y=parameter
    )
    # Monthly variation - all data
    sn.boxplot(ax=ax[1],
               x='Month', y=parameter, data=stations) #, whis=1.5, ax=ax)
    plt.suptitle(f'{parameter}: All sea areas', fontsize=16, y=0.9)
    fig.subplots_adjust(
        left=0.09,
        right=0.95,
        bottom=0.11,
        top=0.85,
        wspace=0.28,
        hspace=0.35
    )
    pdf.savefig(fig)
    fig.savefig(
        os.path.join(save_path, f'{parameter.partition("[")[0]}_all_data.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)
    #########################################################
    # Per area
    #########################################################
    #a=1
    #if a:   # or chose 1! #
    for area, df_area in stations.groupby('Open_sea'):
        #area = 'Gulf of Riga'
        #df_area = stations[stations['Open_sea'] == area].copy()
        print(area)
       # print('Plot map')
        fig = plt.figure(figsize=A4_landscape)
        gs = GridSpec(
            1, 2,
            figure=fig,
            width_ratios=[1.5, 1],
            wspace=0.15,
            top=0.88,  # leave space for title
            bottom=0.08,
            left=0.05,
            right=0.98
        )

        ax_map = fig.add_subplot(gs[0, 0])

        gdf_wgs84.plot(ax=ax_map, rasterized=True)

       # print('Plot positions')
        ax_map.scatter(
            df_area.geometry.x,
            df_area.geometry.y,
            s=8,
            color='red',
            label='Positions',
            rasterized=True
        )

        ax_map.set_title(f'{area}: {parameter}', fontsize=16, pad=0)
        ax_map.legend(fontsize=8)

        # print('Table')
        df_table = df_stats[df_stats['Open_sea'] == area].copy()
        df_table = df_table.drop(columns='Open_sea')
        df_table = df_table.round(2)

        ax_table = fig.add_subplot(gs[0, 1])
        ax_table.axis("off")

        table = ax_table.table(
            cellText=df_table.values.T,
            #colLabels=df_table.index,
            rowLabels=df_table.columns,
            loc="center",
            cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(0.9, 1.4)
        #plt.show()
        pdf.savefig(fig)
        fig.savefig(
            os.path.join(save_path, f'{parameter.partition("[")[0]}_{area.rstrip()}_1.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        # --------------------------------------------------
        # General plots
        # --------------------------------------------------

        fig = plt.figure(figsize=A4_landscape)

        # Main layout: 3 rows x 2 columns
        gs = GridSpec(
            3, 2,
            figure=fig,
            width_ratios=[1, 1.25],
            wspace=0.25,
            hspace=0.35
        )

        # Left column
        ax_hist = fig.add_subplot(gs[0, 0])
        ax_depth = fig.add_subplot(gs[1, 0])
        ax_month = fig.add_subplot(gs[2, 0])

        # Right column: two equally sized scatterplots
        gs_right = GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=gs[:, 1],
            hspace=0.25
        )

        ax1 = fig.add_subplot(gs_right[0])
        ax2 = fig.add_subplot(gs_right[1], sharex=ax1)

        # --------------------------------------------------
        # 1. Histogram
        # --------------------------------------------------

        values = df_area[parameter].dropna().to_numpy()

        ax_hist.hist(
            values,
            bins=50,
            edgecolor='black',
            linewidth=0.5
        )

        # --------------------------------------------------
        # 2. Scatter plots by depth
        # --------------------------------------------------

        depth_intervals = df_area['Depth_int'].dropna().unique()[:2]

        for i, (sub_ax, depth_int) in enumerate(
                zip([ax1, ax2], depth_intervals)
        ):

            d = df_area[
                df_area['Depth_int'] == depth_int
                ].dropna(subset=[parameter])

            if not d.empty:
                sub_ax.scatter(
                    d['datetime'],
                    d[parameter],
                    s=8,
                    color='orange' if i == 1 else None,
                    edgecolors='white',
                    linewidths=0.5,
                    rasterized=True
                )

            # Depth label inside plot
            sub_ax.text(
                0.02, 0.95,
                str(depth_int),
                transform=sub_ax.transAxes,
                fontsize=8,
                va='top',
                ha='left'
            )

            sub_ax.set_axisbelow(True)

            sub_ax.grid(
                True,
                which='major',
                linewidth=0.8,
                alpha=0.5
            )

        # Same datetime range
        xmin = df_area['datetime'].min()
        xmax = df_area['datetime'].max()

        ax1.set_xlim(xmin, xmax)
        ax2.set_xlim(xmin, xmax)

        # Only show datetime labels on lower scatterplot
        ax1.tick_params(axis='x', labelbottom=False)

        # --------------------------------------------------
        # 3. Depth boxplot
        # --------------------------------------------------

        depth_data = []
        depth_labels = []

        for depth_int, d in df_area.groupby(
                'Depth_int',
                observed=True
        ):

            values = d[parameter].dropna().to_numpy()

            if len(values) > 0:
                depth_data.append(values)
                depth_labels.append(str(depth_int))

        if depth_data:
            ax_depth.boxplot(
                depth_data,
                tick_labels=depth_labels
            )

        ax_depth.set_xlabel('Depth (m)')
        ax_depth.set_ylabel(parameter)

        # --------------------------------------------------
        # 4. Monthly boxplot
        # --------------------------------------------------

        month_data = []
        month_labels = []

        for month, d in df_area.groupby('Month'):

            values = d[parameter].dropna().to_numpy()

            if len(values) > 0:
                month_data.append(values)
                month_labels.append(month)

        if month_data:
            ax_month.boxplot(
                month_data,
                tick_labels=month_labels
            )

        ax_month.set_xlabel('Month')
        ax_month.set_ylabel(parameter)

        # --------------------------------------------------
        # Figure title
        # --------------------------------------------------

        fig.suptitle(
            f'{area}: {parameter}',
            fontsize=16,
            y=0.96
        )

        # --------------------------------------------------
        # Layout
        # --------------------------------------------------

        fig.subplots_adjust(
            left=0.09,
            right=0.95,
            bottom=0.10,
            top=0.90
        )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        pdf.savefig(fig)

        fig.savefig(
            os.path.join(
                save_path,
                f'{parameter.partition("[")[0]}_{area.rstrip()}_2.png'
            ),
            dpi=300,
            bbox_inches='tight'
        )

        plt.close(fig)

    pdf.close()
print('That is all')