import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage import morphology, filters
from PIL import Image, ImageFilter
from decimal import Decimal
from skimage import restoration


def plot_overlapping_tracks(image_name, rojo_df, verde_df, file_results, plots_directory):
    for result in file_results:
        rojo_track = rojo_df[rojo_df['TRACK_ID'] == result['Rojo_TRACK_ID']]
        verde_track = verde_df[verde_df['TRACK_ID'] == result['Verde_TRACK_ID']]

        # Plot MEAN_INTENSITY_CH1 vs FRAME for Rojo with background intensity values
        plt.figure()
        plt.plot(rojo_track['FRAME'], rojo_track['MEAN_INTENSITY_CH1'], 'r-', label=f'Rojo {result["Rojo_TRACK_ID"]}')
        if 'Intensity-Bg Rojo' in rojo_track.columns:
            plt.plot(rojo_track['FRAME'], rojo_track['Intensity-Bg Rojo'], 'r--', label='Intensity-Bg Rojo')
        plt.xlabel('FRAME')
        plt.ylabel('MEAN_INTENSITY_CH1 / Intensity-Bg')
        plt.title(f'{image_name} - Rojo MEAN_INTENSITY_CH1 vs FRAME with Intensity-Bg')
        plt.legend()
        plt.savefig(os.path.join(plots_directory,
                                 f'{image_name}_Rojo_MEAN_INTENSITY_CH1_vs_FRAME_with_Intensity-Bg_{result["Rojo_TRACK_ID"]}.png'))
        plt.close()

        # Plot MEAN_INTENSITY_CH1 vs FRAME for Verde with background intensity values
        plt.figure()
        plt.plot(verde_track['FRAME'], verde_track['MEAN_INTENSITY_CH1'], 'g-',
                 label=f'Verde {result["Verde_TRACK_ID"]}')
        if 'Intensity-Bg Verde' in verde_track.columns:
            plt.plot(verde_track['FRAME'], verde_track['Intensity-Bg Verde'], 'g--', label='Intensity-Bg Verde')
        plt.xlabel('FRAME')
        plt.ylabel('MEAN_INTENSITY_CH1 / Intensity-Bg')
        plt.title(f'{image_name} - Verde MEAN_INTENSITY_CH1 vs FRAME with Intensity-Bg')
        plt.legend()
        plt.savefig(os.path.join(plots_directory,
                                 f'{image_name}_Verde_MEAN_INTENSITY_CH1_vs_FRAME_with_Intensity-Bg_{result["Verde_TRACK_ID"]}.png'))
        plt.close()


def calculate_msd(trajectory, max_lag):
    msd = np.zeros(max_lag)
    n = len(trajectory)
    for lag in range(1, max_lag + 1):
        displacements = trajectory[lag:] - trajectory[:-lag]
        squared_displacements = np.sum(displacements ** 2, axis=1)
        msd[lag - 1] = np.mean(squared_displacements)
    return msd


def calculate_alpha(msd, time_lags):
    log_msd = np.log(msd)
    log_time_lags = np.log(time_lags)
    alpha, _ = np.polyfit(log_time_lags, log_msd, 1)
    return alpha


def calculate_diffusion_coefficient(msd, time_lags):
    D = msd / (2 * time_lags)
    return D


def calculate_sMSS(trajectory, max_lag):
    msd = calculate_msd(trajectory, max_lag)
    sMSS = np.std(msd)
    return sMSS


def calculate_mean(slice_np, roi):
    mean_value = np.mean(slice_np[roi])
    return mean_value


def find_csv(input_directory):
    rojo_csv_files_spots = {}
    verde_csv_files_spots = {}
    rojo_csv_files_tracks = {}
    verde_csv_files_tracks = {}

    # Iterate through the main directories
    for main_dir in os.listdir(input_directory):
        if 'rojo' in main_dir or 'verde' in main_dir:
            main_path = os.path.join(input_directory, main_dir)
            if not os.path.exists(main_path):
                continue
            # Iterate through the subdirectories
            for sub_dir in os.listdir(main_path):
                if 'Moving' in sub_dir or 'Fixed' in sub_dir or 'moving' in sub_dir or 'fixed' in sub_dir:
                    sub_path = os.path.join(main_path, sub_dir)
                    # Check if the directory contains 'SPT_Analysis'
                    spt_analysis_path = os.path.join(sub_path, 'SPT_Analysis')
                    if os.path.exists(spt_analysis_path):
                        # Find CSV files containing specific strings
                        for file_name in os.listdir(spt_analysis_path):
                            if '_Spots in tracks statistics.csv' in file_name:
                                image_name = file_name.replace('_Spots in tracks statistics.csv', '').replace('fixed',
                                                                                                              '').replace(
                                    'Fixed', '').replace('moving', '').replace('Moving', '').strip()
                                if 'rojo' in main_dir:
                                    rojo_csv_files_spots[image_name] = os.path.join(spt_analysis_path, file_name)
                                elif 'verde' in main_dir:
                                    verde_csv_files_spots[image_name] = os.path.join(spt_analysis_path, file_name)
                            elif '_Tracks statistics.csv' in file_name:
                                image_name = file_name.replace('_Tracks statistics.csv', '').replace('fixed',
                                                                                                     '').replace(
                                    'Fixed', '').replace('moving', '').replace('Moving', '').strip()
                                if 'rojo' in main_dir:
                                    rojo_csv_files_tracks[image_name] = os.path.join(spt_analysis_path, file_name)
                                elif 'verde' in main_dir:
                                    verde_csv_files_tracks[image_name] = os.path.join(spt_analysis_path, file_name)

    return rojo_csv_files_spots, verde_csv_files_spots, rojo_csv_files_tracks, verde_csv_files_tracks


def analyze_trajectories(image_name, rojo_file, verde_file, distancia_x, distancia_y, min_frames_overlap, frame_gap,
                         spot_range_min, spot_range_max):
    print(f"Analyzing {image_name} - Rojo: {rojo_file} \n Verde: {verde_file}")

    rojo_df = pd.read_csv(rojo_file)
    verde_df = pd.read_csv(verde_file)

    # Process images and get intensity values
    spotIntensityBg_rojo = rojo_df['Intensity-Bg Subtract']
    slicesIntensityBg_rojo = rojo_df['MEAN_INTENSITY_CH1'] - rojo_df['Intensity-Bg Subtract']
    spotIntensityBg_verde = verde_df['Intensity-Bg Subtract']
    slicesIntensityBg_verde = verde_df['MEAN_INTENSITY_CH1'] - verde_df['Intensity-Bg Subtract']

    file_results = []

    for rojo_track_id in rojo_df['TRACK_ID'].unique():
        rojo_track = rojo_df[rojo_df['TRACK_ID'] == rojo_track_id]
        for verde_track_id in verde_df['TRACK_ID'].unique():
            verde_track = verde_df[verde_df['TRACK_ID'] == verde_track_id]
            overlap_start_frame_rojo = None
            overlap_end_frame_rojo = None
            overlap_start_frame_verde = None
            overlap_end_frame_verde = None

            for _, rojo_row in rojo_track.iterrows():
                for _, verde_row in verde_track.iterrows():
                    if (abs(rojo_row['POSITION_X'] - verde_row['POSITION_X']) <= distancia_x and
                            abs(rojo_row['POSITION_Y'] - verde_row['POSITION_Y']) <= distancia_y and
                            abs(rojo_row['FRAME'] - verde_row['FRAME']) <= frame_gap):
                        if overlap_start_frame_rojo is None or rojo_row['FRAME'] < overlap_start_frame_rojo:
                            overlap_start_frame_rojo = rojo_row['FRAME']
                        if overlap_start_frame_verde is None or verde_row['FRAME'] < overlap_start_frame_verde:
                            overlap_start_frame_verde = verde_row['FRAME']
                        if overlap_end_frame_rojo is None or rojo_row['FRAME'] > overlap_end_frame_rojo:
                            overlap_end_frame_rojo = rojo_row['FRAME']
                        if overlap_end_frame_verde is None or verde_row['FRAME'] > overlap_end_frame_verde:
                            overlap_end_frame_verde = verde_row['FRAME']

            if overlap_start_frame_rojo is not None and overlap_start_frame_verde is not None:
                overlap_frames_rojo = overlap_end_frame_rojo - overlap_start_frame_rojo + 1
                overlap_frames_verde = overlap_end_frame_verde - overlap_start_frame_verde + 1

                if overlap_frames_rojo >= min_frames_overlap and overlap_frames_verde >= min_frames_overlap:
                    rojo_trajectory = rojo_track[['POSITION_X', 'POSITION_Y']].values
                    verde_trajectory = verde_track[['POSITION_X', 'POSITION_Y']].values

                    max_lag = 4
                    time_lags = np.arange(1, max_lag + 1)

                    msd_rojo = abs(calculate_msd(rojo_trajectory, max_lag))
                    msd_verde = abs(calculate_msd(verde_trajectory, max_lag))
                    alpha_rojo = abs(calculate_alpha(msd_rojo, time_lags))
                    alpha_verde = abs(calculate_alpha(msd_verde, time_lags))
                    sMSS_rojo = abs(calculate_sMSS(rojo_trajectory, max_lag))
                    sMSS_verde = abs(calculate_sMSS(verde_trajectory, max_lag))
                    # Calculate Diffusion Coefficient (D) as the slope of the linear fitting of the first time lag of the MSD curve
                    D_rojo_slope, _ = abs(np.polyfit(time_lags[:1], msd_rojo[:1], 1))
                    D_verde_slope, _ = abs(np.polyfit(time_lags[:1], msd_verde[:1], 1))
                    # Calculate Short-Time Lag Diffusion Coefficient (D1-4) as the slope of the linear fitting of the first 4 time lags of the MSD curve
                    D1_4_rojo_slope, _ = abs(np.polyfit(time_lags[:4], msd_rojo[:4], 1))
                    D1_4_verde_slope, _ = abs(np.polyfit(time_lags[:4], msd_verde[:4], 1))
                    # Calculate MSD for all time lags of the overlapping trajectory
                    msd_all_rojo = abs(calculate_msd(rojo_trajectory, len(rojo_trajectory) - 1))
                    msd_all_verde = abs(calculate_msd(verde_trajectory, len(verde_trajectory) - 1))
                    # Calculate mean intensity in the specified spot range
                    rojo_intensity_range = rojo_track[
                        (rojo_track['FRAME'] >= overlap_start_frame_rojo + spot_range_min) & (
                                rojo_track['FRAME'] <= overlap_start_frame_rojo + spot_range_max)]
                    verde_intensity_range = verde_track[
                        (verde_track['FRAME'] >= overlap_start_frame_verde + spot_range_min) & (
                                verde_track['FRAME'] <= overlap_start_frame_verde + spot_range_max)]
                    if len(rojo_intensity_range) == 0:
                        rojo_intensity_range = rojo_track
                    if len(verde_intensity_range) == 0:
                        verde_intensity_range = verde_track
                    # Find the column with the pattern 'MEAN_INTENSITY_'
                    rojo_intensity_column = [col for col in rojo_intensity_range.columns if 'MEAN_INTENSITY_' in col][0]
                    verde_intensity_column = [col for col in verde_intensity_range.columns if 'MEAN_INTENSITY_' in col][
                        0]
                    mean_intensity_rojo = rojo_intensity_range[rojo_intensity_column].mean()
                    mean_intensity_verde = verde_intensity_range[verde_intensity_column].mean()
                    # Calcula los valores promedio solo para las trayectorias solapantes
                    intensity_bg_subtract_rojo = rojo_track['MEAN_INTENSITY_CH1'] - rojo_track['Intensity-Bg Subtract']
                    intensity_bg_subtract_range_rojo = intensity_bg_subtract_rojo[
                                                       spot_range_min:spot_range_max + 1].mean()
                    intensity_bg_subtract_verde = verde_track['MEAN_INTENSITY_CH1'] - verde_track[
                        'Intensity-Bg Subtract']
                    intensity_bg_subtract_range_verde = intensity_bg_subtract_verde[
                                                        spot_range_min:spot_range_max + 1].mean()
                    spot_intensity_bg_subtract_rojo = rojo_track['Intensity-Bg Subtract']
                    spot_intensity_bg_subtract_range_rojo = spot_intensity_bg_subtract_rojo[
                                                            spot_range_min:spot_range_max + 1].mean()
                    spot_intensity_bg_subtract_verde = verde_track['Intensity-Bg Subtract']
                    spot_intensity_bg_subtract_range_verde = spot_intensity_bg_subtract_verde[
                                                             spot_range_min:spot_range_max + 1].mean()
                    # Determine if the trajectory is long or short based on the length threshold
                    rojo_length = len(rojo_track)
                    verde_length = len(verde_track)
                    rojo_length_category = 'Long' if rojo_length >= length_threshold else 'Short'
                    verde_length_category = 'Long' if verde_length >= length_threshold else 'Short'
                    # Determine motility based on the motility threshold
                    rojo_motility = 'Mobile' if D1_4_rojo_slope >= motility_threshold else 'Inmobile'
                    verde_motility = 'Mobile' if D1_4_verde_slope >= motility_threshold else 'Inmobile'
                    # Determine movement type based on sMSS values
                    if sMSS_rojo == 1.0:
                        rojo_movement = 'Unidirectional Ballistic'
                    elif sMSS_rojo == 0:
                        rojo_movement = 'Immobility'
                    elif 0.45 <= sMSS_rojo <= 0.55:
                        rojo_movement = 'Free'
                    elif 0 < sMSS_rojo < 0.45:
                        rojo_movement = 'Confined'
                    elif sMSS_rojo > 0.55:
                        rojo_movement = 'Directed'
                    else:
                        rojo_movement = 'Undefined'
                    if sMSS_verde == 1.0:
                        verde_movement = 'Unidirectional Ballistic'
                    elif sMSS_verde == 0:
                        verde_movement = 'Immobility'
                    elif 0.45 <= sMSS_verde <= 0.55:
                        verde_movement = 'Free'
                    elif 0 < sMSS_verde < 0.45:
                        verde_movement = 'Confined'
                    elif sMSS_verde > 0.55:
                        verde_movement = 'Directed'
                    else:
                        verde_movement = 'Undefined'
                    # Determine movement type based on alpha values
                    if 0 <= alpha_rojo < 0.6:
                        alpha_movement_rojo = 'Confined'
                    elif 0.6 <= alpha_rojo < 0.9:
                        alpha_movement_rojo = 'Anomalous'
                    elif 0.9 <= alpha_rojo < 1.1:
                        alpha_movement_rojo = 'Free'
                    elif alpha_rojo >= 1.1:
                        alpha_movement_rojo = 'Directed'
                    else:
                        alpha_movement_rojo = 'Undefined'
                    if 0 <= alpha_verde < 0.6:
                        alpha_movement_verde = 'Confined'
                    elif 0.6 <= alpha_verde < 0.9:
                        alpha_movement_verde = 'Anomalous'
                    elif 0.9 <= alpha_verde < 1.1:
                        alpha_movement_verde = 'Free'
                    elif alpha_verde >= 1.1:
                        alpha_movement_verde = 'Directed'
                    else:
                        alpha_movement_verde = 'Undefined'
                    file_results.append({
                        'Image Title': image_name,
                        'Rojo_TRACK_ID': rojo_track_id,
                        'Verde_TRACK_ID': verde_track_id,
                        'Overlap_Frames_Rojo': overlap_frames_rojo,
                        'Overlap_Frames_Verde': overlap_frames_verde,
                        'Overlap_Start_Frame_Rojo': overlap_start_frame_rojo,
                        'Overlap_End_Frame_Rojo': overlap_end_frame_rojo,
                        'Overlap_Start_Frame_Verde': overlap_start_frame_verde,
                        'Overlap_End_Frame_Verde': overlap_end_frame_verde,
                        'Absolute_Overlap_Frames_Rojo': len(rojo_track),
                        'Absolute_Overlap_Frames_Verde': len(verde_track),
                        'MSD Time Lag 1 Rojo': msd_rojo[0],
                        'MSD Time Lag 2 Rojo': msd_rojo[1],
                        'MSD Time Lag 3 Rojo': msd_rojo[2],
                        'MSD Time Lag 1 Verde': msd_verde[0],
                        'MSD Time Lag 2 Verde': msd_verde[1],
                        'MSD Time Lag 3 Verde': msd_verde[2],
                        'MSD Rojo': msd_rojo[3],
                        'MSD Verde': msd_verde[3],
                        'Diffusion Coefficient Rojo': D_rojo_slope,
                        'Diffusion Coefficient Verde': D_verde_slope,
                        'Alpha Rojo': alpha_rojo,
                        'Alpha Verde': alpha_verde,
                        'Alpha Movement Rojo': alpha_movement_rojo,
                        'Alpha Movement Verde': alpha_movement_verde,
                        'sMSS Rojo': sMSS_rojo,
                        'sMSS Verde': sMSS_verde,
                        'sMSS Rojo Movement': rojo_movement,
                        'sMSS Verde Movement': verde_movement,
                        'Short-Time Lag Diffusion Coefficient Rojo (D1-4)': D1_4_rojo_slope,
                        'Short-Time Lag Diffusion Coefficient Verde (D1-4)': D1_4_verde_slope,
                        'Track Mean Intensity in Spot Range (' + str(spot_range_min) + '-' + str(
                            spot_range_max) + ') Rojo': mean_intensity_rojo,
                        'Track Mean Intensity in Spot Range (' + str(spot_range_min) + '-' + str(
                            spot_range_max) + ') Verde': mean_intensity_verde,
                        'Track Length Rojo': rojo_length_category,
                        'Track Length Verde': verde_length_category,
                        'Motility Rojo': rojo_motility,
                        'Motility Verde': verde_motility,
                        'Spot Intensity-Bg Rojo': spot_intensity_bg_subtract_rojo.mean(),
                        'Intensity-Bg Rojo': intensity_bg_subtract_rojo.mean(),
                        'Spot Intensity-Bg (' + str(spot_range_min) + '-' + str(
                            spot_range_max) + ') Rojo': spot_intensity_bg_subtract_range_rojo,
                        'Intensity-Bg (' + str(spot_range_min) + '-' + str(
                            spot_range_max) + ') Rojo': intensity_bg_subtract_range_rojo,
                        'Spot Intensity-Bg Verde': spot_intensity_bg_subtract_verde.mean(),
                        'Intensity-Bg Verde': intensity_bg_subtract_verde.mean(),
                        'Spot Intensity-Bg (' + str(spot_range_min) + '-' + str(
                            spot_range_max) + ') Verde': spot_intensity_bg_subtract_range_verde,
                        'Intensity-Bg (' + str(spot_range_min) + '-' + str(
                            spot_range_max) + ') Verde': intensity_bg_subtract_range_verde

                    })
    return image_name, file_results, rojo_df, verde_df


# Ask the user for the input directory
input_directory = input("Please enter the input directory: ")

# Ask the user for additional parameters with default values
distancia_x = float(input("Please enter the distance in X (default 0.4): ") or 0.4)
distancia_y = float(input("Please enter the distance in Y (default 0.4): ") or 0.4)
min_frames_overlap = int(
    input("Please enter the minimum number of overlapping frames in the trajectory (default 5): ") or 5)
frame_gap = int(input("Please enter the time gap (default 0): ") or 0)
spot_range = input("Please enter the spot range (e.g., 0-19) (default 0-19): ") or "0-19"
spot_range_min, spot_range_max = map(int, spot_range.split('-'))
length_threshold = int(input("Please enter the length threshold for long tracks (default 30): ") or 30)
motility_threshold = float(input("Please enter the D1-4 motility threshold (default 0.002): ") or 0.002)
# Get the column indexes for summary from the user
column_indexes_no_overlap_input = input(
    "Get the column indexes (numeric numbers from 0) for No Overlapping summary (default 33,35): ")
column_indexes_overlap_input_rojo = input("Get the column indexes for Overlapping summary ROJO (default 33,35): ")
column_indexes_overlap_input_verde = input("Get the column indexes for Overlapping summary VERDE (default 33,35): ")


if column_indexes_no_overlap_input:
    column_indexes_no_overlap = list(map(int, column_indexes_no_overlap_input.split(',')))
else:
    column_indexes_no_overlap = [33, 35]

if column_indexes_overlap_input_rojo:
    column_indexes_overlap_rojo = list(map(int, column_indexes_overlap_input_rojo.split(',')))
else:
    column_indexes_overlap_rojo = [33, 35]

if column_indexes_overlap_input_verde:
    column_indexes_overlap_verde = list(map(int, column_indexes_overlap_input_verde.split(',')))
else:
    column_indexes_overlap_verde = [33, 35]


# Find the CSV and TIF files
rojo_csv_files_spots, verde_csv_files_spots, rojo_csv_files_tracks, verde_csv_files_tracks = find_csv(
    input_directory)

# Print keys for debugging
# print("Keys in rojo_csv_files_spots:", list(rojo_csv_files_spots.keys()))
# print("Keys in rojo_tif_files:", list(rojo_tif_files.keys()))

# Analyze the trajectories in parallel and get the results
results_spots = {}
rojo_dfs_spots = {}
verde_dfs_spots = {}
with ThreadPoolExecutor() as executor:
    futures_spots = [
        executor.submit(analyze_trajectories, image_name, rojo_csv_files_spots[image_name],
                        verde_csv_files_spots[image_name],
                        distancia_x, distancia_y, min_frames_overlap, frame_gap, spot_range_min, spot_range_max)
        for image_name in rojo_csv_files_spots if image_name in verde_csv_files_spots
    ]
    for future in futures_spots:
        image_name, file_results, rojo_df, verde_df = future.result()
        results_spots[image_name] = file_results
        rojo_dfs_spots[image_name] = rojo_df
        verde_dfs_spots[image_name] = verde_df

# Create the results_dualTIRFM directory in input_directory and rewrite if already exists
results_directory = os.path.join(input_directory, 'results_dualTIRFM')
if os.path.exists(results_directory):
    import shutil

    shutil.rmtree(results_directory)
os.makedirs(results_directory)

# Create the 'Summary_Analysis' directory in the results_directory
summary_analysis_directory = os.path.join(results_directory, 'Summary_Analysis')
os.makedirs(summary_analysis_directory, exist_ok=True)
print(f"Directory 'Summary_Analysis' created successfully in {results_directory}.")


def extract_column_values_single_file(column_indexes_no_overlap, column_indexes_overlap_rojo, column_indexes_overlap_verde):

    column_values_dict_rojo_no_overlap = {index: [] for index in column_indexes_no_overlap}
    column_values_dict_verde_no_overlap = {index: [] for index in column_indexes_no_overlap}
    column_values_dict_rojo_overlap = {index: [] for index in column_indexes_overlap_rojo}
    column_values_dict_verde_overlap = {index: [] for index in column_indexes_overlap_verde}

    image_names = []

    for image_name in rojo_csv_files_tracks:
        if image_name in verde_csv_files_tracks:

            csv_directory = os.path.join(results_directory, image_name, "csv")
            rojo_tracks_df = pd.read_csv(rojo_csv_files_tracks[image_name])
            verde_tracks_df = pd.read_csv(verde_csv_files_tracks[image_name])

            for index in column_indexes_no_overlap:
                column_values_dict_rojo_no_overlap[index].append(rojo_tracks_df.iloc[:, index].values.tolist())
                column_values_dict_verde_no_overlap[index].append(verde_tracks_df.iloc[:, index].values.tolist())

            results_file_path = os.path.join(csv_directory, f'{image_name}_recalculated_trajectory_overlap_results.csv')
            results_df = pd.read_csv(results_file_path)

            # Para ROJO
            for index in column_indexes_overlap_rojo:
                column_values_dict_rojo_overlap[index].append(results_df.iloc[:, index].values.tolist())

            # Para VERDE
            for index in column_indexes_overlap_verde:
                column_values_dict_verde_overlap[index].append(results_df.iloc[:, index].values.tolist())

            image_names.append(image_name)


    # Guardar valores de cada columna en archivos CSV separados
    for index in column_indexes_no_overlap:
        column_name = rojo_tracks_df.columns[index]
        # Rojo No Overlap
        output_file_rojo_no_overlap = os.path.join(summary_analysis_directory,
                                                   f'{column_name}_rojo_No_overlap_Track_statistics.csv')
        df_rojo_no_overlap = pd.DataFrame(column_values_dict_rojo_no_overlap[index]).transpose()
        df_rojo_no_overlap.columns = image_names
        df_rojo_no_overlap.to_csv(output_file_rojo_no_overlap, index=False)

        # Verde No Overlap
        output_file_verde_no_overlap = os.path.join(summary_analysis_directory,
                                                    f'{column_name}_verde_No_overlap_Track_statistics.csv')
        df_verde_no_overlap = pd.DataFrame(column_values_dict_verde_no_overlap[index]).transpose()
        df_verde_no_overlap.columns = image_names
        df_verde_no_overlap.to_csv(output_file_verde_no_overlap, index=False)

    # Rojo Overlap
    for index in column_indexes_overlap_rojo:
        column_name = results_df.columns[index]
        output_file_rojo_overlap = os.path.join(summary_analysis_directory,
                                                f'{column_name}_rojo_overlap_Track_statistics.csv')
        df_rojo_overlap = pd.DataFrame(column_values_dict_rojo_overlap[index]).transpose()
        df_rojo_overlap.columns = image_names
        df_rojo_overlap.to_csv(output_file_rojo_overlap, index=False)

    # Verde Overlap
    for index in column_indexes_overlap_verde:
        column_name = results_df.columns[index]
        output_file_verde_overlap = os.path.join(summary_analysis_directory,
                                                 f'{column_name}_verde_overlap_Track_statistics.csv')
        df_verde_overlap = pd.DataFrame(column_values_dict_verde_overlap[index]).transpose()
        df_verde_overlap.columns = image_names
        df_verde_overlap.to_csv(output_file_verde_overlap, index=False)

    # Guardar también archivos con una sola columna y filas vacías entre imágenes
    for index in column_indexes_no_overlap:
        column_name = rojo_tracks_df.columns[index]

        # Rojo No Overlap - 1 columna
        output_file_rojo_no_overlap_1col = os.path.join(summary_analysis_directory,
            f'{column_name}_rojo_No_overlap_Track_statistics_1column.csv')
        values_rojo = [item for sublist in column_values_dict_rojo_no_overlap[index] for item in sublist + [""]]
        pd.DataFrame(values_rojo, columns=[column_name]).to_csv(output_file_rojo_no_overlap_1col, index=False)

        # Verde No Overlap - 1 columna
        output_file_verde_no_overlap_1col = os.path.join(summary_analysis_directory,
            f'{column_name}_verde_No_overlap_Track_statistics_1column.csv')
        values_verde = [item for sublist in column_values_dict_verde_no_overlap[index] for item in sublist + [""]]
        pd.DataFrame(values_verde, columns=[column_name]).to_csv(output_file_verde_no_overlap_1col, index=False)

    # Rojo Overlap - 1 columna
    for index in column_indexes_overlap_rojo:
        column_name = results_df.columns[index]
        output_file_rojo_overlap_1col = os.path.join(summary_analysis_directory,
                                                     f'{column_name}_rojo_overlap_Track_statistics_1column.csv')
        values_rojo = [item for sublist in column_values_dict_rojo_overlap[index] for item in sublist + [""]]
        pd.DataFrame(values_rojo, columns=[column_name]).to_csv(output_file_rojo_overlap_1col, index=False)

    # Verde Overlap - 1 columna
    for index in column_indexes_overlap_verde:
        column_name = results_df.columns[index]
        output_file_verde_overlap_1col = os.path.join(summary_analysis_directory,
                                                      f'{column_name}_verde_overlap_Track_statistics_1column.csv')
        values_verde = [item for sublist in column_values_dict_verde_overlap[index] for item in sublist + [""]]
        pd.DataFrame(values_verde, columns=[column_name]).to_csv(output_file_verde_overlap_1col, index=False)


# Save each result to a separate CSV file with the image name before 'trajectory_overlap_results.csv'
summary_data = []
for image_name, file_results in results_spots.items():
    image_directory = os.path.join(results_directory, image_name)
    os.makedirs(image_directory, exist_ok=True)

    # Create a directory named "csv" in the image directory if it doesn't exist
    csv_directory = os.path.join(image_directory, "csv")
    os.makedirs(csv_directory, exist_ok=True)

    # Create a directory named "plots" in the image directory if it doesn't exist
    plots_directory = os.path.join(image_directory, "plots")
    os.makedirs(plots_directory, exist_ok=True)

    # Guardar archivo general
    results_df = pd.DataFrame(file_results)
    results_file_path = os.path.join(csv_directory, f'{image_name}_recalculated_trajectory_overlap_results.csv')
    results_df.to_csv(results_file_path, index=False)

    # Crear y guardar archivo solo con columnas 'Rojo' + 'Image Title'
    rojo_columns = [col for col in results_df.columns if 'Rojo' in col or col == 'Image Title']
    results_df_rojo = results_df[rojo_columns]
    results_file_path_rojo = os.path.join(csv_directory,
                                          f'{image_name}_rojo_recalculated_trajectory_overlap_results.csv')
    results_df_rojo.to_csv(results_file_path_rojo, index=False)

    # Crear y guardar archivo solo con columnas 'Verde' + 'Image Title'
    verde_columns = [col for col in results_df.columns if 'Verde' in col or col == 'Image Title']
    results_df_verde = results_df[verde_columns]
    results_file_path_verde = os.path.join(csv_directory,
                                           f'{image_name}_verde_recalculated_trajectory_overlap_results.csv')
    results_df_verde.to_csv(results_file_path_verde, index=False)

    # Collect summary data per image
    n_overlapping_tracks = len(file_results)
    n_red_tracks_overlapping = len(set([result['Rojo_TRACK_ID'] for result in file_results]))
    n_green_tracks_overlapping = len(set([result['Verde_TRACK_ID'] for result in file_results]))

    # Calculate non-overlapping tracks
    total_red_tracks = len(rojo_dfs_spots[image_name]['TRACK_ID'].unique())
    total_green_tracks = len(verde_dfs_spots[image_name]['TRACK_ID'].unique())
    n_red_tracks_no_overlapping = total_red_tracks - n_red_tracks_overlapping
    n_green_tracks_no_overlapping = total_green_tracks - n_green_tracks_overlapping
    n_no_overlapping_tracks = n_red_tracks_no_overlapping + n_green_tracks_no_overlapping

    summary_data.append({
        'Image Title': image_name,
        'N of Overlapping Tracks': n_overlapping_tracks,
        'N of Red Tracks overlapping': n_red_tracks_overlapping,
        'N of Green Tracks overlapping': n_green_tracks_overlapping,
        'N of No Overlapping Tracks': n_no_overlapping_tracks,
        'N of Red Tracks No Overlapping': n_red_tracks_no_overlapping,
        'N of Green Tracks No Overlapping': n_green_tracks_no_overlapping
    })

    # Generate plots for overlapping tracks
    rojo_df = rojo_dfs_spots[image_name]
    verde_df = verde_dfs_spots[image_name]
    # plot_overlapping_tracks(image_name, rojo_df, verde_df, file_results, plots_directory)

    for result in file_results:
        rojo_track = rojo_df[rojo_df['TRACK_ID'] == result['Rojo_TRACK_ID']]
        verde_track = verde_df[verde_df['TRACK_ID'] == result['Verde_TRACK_ID']]

        # Plot POSITION_X vs FRAME
        plt.figure()
        plt.plot(rojo_track['FRAME'], rojo_track['POSITION_X'], 'r-', label=f'Rojo {result["Rojo_TRACK_ID"]}')
        plt.plot(verde_track['FRAME'], verde_track['POSITION_X'], 'g-', label=f'Verde {result["Verde_TRACK_ID"]}')
        plt.xlabel('FRAME')
        plt.ylabel('POSITION_X')
        plt.ylim([rojo_track['POSITION_X'].min() - 2, verde_track['POSITION_X'].max() + 2])  # Adjust y-axis limits
        plt.title(f'{image_name} - POSITION_X vs FRAME')
        plt.legend()
        plt.savefig(os.path.join(plots_directory,
                                 f'{image_name}_POSITION_X_vs_FRAME_{result["Rojo_TRACK_ID"]}_{result["Verde_TRACK_ID"]}.png'))
        plt.close()

        # Plot POSITION_Y vs FRAME
        plt.figure()
        plt.plot(rojo_track['FRAME'], rojo_track['POSITION_Y'], 'r-', label=f'Rojo {result["Rojo_TRACK_ID"]}')
        plt.plot(verde_track['FRAME'], verde_track['POSITION_Y'], 'g-', label=f'Verde {result["Verde_TRACK_ID"]}')
        plt.xlabel('FRAME')
        plt.ylabel('POSITION_Y')
        plt.ylim([rojo_track['POSITION_Y'].min() - 2, verde_track['POSITION_Y'].max() + 2])  # Adjust y-axis limits
        plt.title(f'{image_name} - POSITION_Y vs FRAME')
        plt.legend()
        plt.savefig(os.path.join(plots_directory,
                                 f'{image_name}_POSITION_Y_vs_FRAME_{result["Rojo_TRACK_ID"]}_{result["Verde_TRACK_ID"]}.png'))
        plt.close()

        # # Create CSV file with POSITION_X, POSITION_Y, and FRAME
        # combined_df = pd.DataFrame({
        #     'Rojo_POSITION_X': rojo_track['POSITION_X'],
        #     'Rojo_POSITION_Y': rojo_track['POSITION_Y'],
        #     'Rojo_FRAME': rojo_track['FRAME'],
        #     'Verde_POSITION_X': verde_track['POSITION_X'],
        #     'Verde_POSITION_Y': verde_track['POSITION_Y'],
        #     'Verde_FRAME': verde_track['FRAME']
        # })
        # csv_filename = f'{image_name}_POSITION_XY_vs_FRAME_{result["Rojo_TRACK_ID"]}_{result["Verde_TRACK_ID"]}.csv'
        # combined_df.to_csv(os.path.join(plots_directory, csv_filename), index=False)

        # Perform a full outer join on FRAME to include all rows
        combined_df = pd.merge(rojo_track[['POSITION_X', 'POSITION_Y', 'FRAME']],
                               verde_track[['POSITION_X', 'POSITION_Y', 'FRAME']], on='FRAME', how='outer',
                               suffixes=('_Rojo', '_Verde'))

        # Create CSV file with POSITION_X, POSITION_Y, and FRAME
        csv_filename = f'{image_name}_POSITION_XY_vs_FRAME_{result["Rojo_TRACK_ID"]}_{result["Verde_TRACK_ID"]}.csv'
        combined_df.to_csv(os.path.join(plots_directory, csv_filename), index=False)

        # # Plot 3D scatter plot POSITION_X, POSITION_Y, FRAME
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(rojo_track['POSITION_X'], rojo_track['POSITION_Y'], rojo_track['FRAME'], 'r-', label='Rojo')
        # ax.plot(verde_track['POSITION_X'], verde_track['POSITION_Y'], verde_track['FRAME'], 'g-', label='Verde')
        # ax.set_xlabel('POSITION_X')
        # ax.set_ylabel('POSITION_Y')
        # ax.set_zlabel('FRAME')
        # ax.set_title(f'{image_name}_POSITION_XY_vs_FRAME')
        # plt.savefig(os.path.join(plots_directory,
        #                          f'{image_name}_POSITION_XY_vs_FRAME_{result["Rojo_TRACK_ID"]}_{result["Verde_TRACK_ID"]}.png'))
        # plt.close()

# Save summary data to a CSV file
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(summary_analysis_directory, 'summary_trajectory_overlap_results.csv'), index=False)


# Function to update the CSV spot_in tracks statistics files
def update_csv(df, file_name):
    # Calculate 'Intensity-Bg Rojo' and 'Intensity-Bg Verde'
    if 'rojo' in file_name:
        df['Intensity-Bg Rojo'] = df['MEAN_INTENSITY_CH1'] - df['Intensity-Bg Subtract']
    elif 'verde' in file_name:
        df['Intensity-Bg Verde'] = df['MEAN_INTENSITY_CH1'] - df['Intensity-Bg Subtract']

    # Rename 'Intensity-Bg Subtract' to 'Spot Intensity-Bg Subtract'
    df.rename(columns={'Intensity-Bg Subtract': 'Spot Intensity-Bg Subtract'}, inplace=True)
    # Remove the 'Unnamed: 21' column if it exists
    if 'Unnamed: 21' in df.columns:
        df.drop(columns=['Unnamed: 21'], inplace=True)
    return df


def calculate_average_values(df, track_id_column, columns_to_average):
    average_values = df.groupby(track_id_column)[columns_to_average].mean().reset_index()
    return average_values


# Process _Spots in tracks statistics.csv files
for image_name in rojo_csv_files_spots:
    if image_name in verde_csv_files_spots:
        rojo_spots_df = pd.read_csv(rojo_csv_files_spots[image_name])
        verde_spots_df = pd.read_csv(verde_csv_files_spots[image_name])
        overlapping_rojo_ids = set(result['Rojo_TRACK_ID'] for result in results_spots[image_name])
        overlapping_verde_ids = set(result['Verde_TRACK_ID'] for result in results_spots[image_name])
        rojo_overlap_df = rojo_spots_df[rojo_spots_df['TRACK_ID'].isin(overlapping_rojo_ids)]
        rojo_no_overlap_df = rojo_spots_df[~rojo_spots_df['TRACK_ID'].isin(overlapping_rojo_ids)]
        verde_overlap_df = verde_spots_df[verde_spots_df['TRACK_ID'].isin(overlapping_verde_ids)]
        verde_no_overlap_df = verde_spots_df[~verde_spots_df['TRACK_ID'].isin(overlapping_verde_ids)]

        # Update the dataframes with new columns and renamed column
        rojo_overlap_df = update_csv(rojo_overlap_df, f'{image_name}_rojo_overlap_Spots_statistics.csv')
        rojo_no_overlap_df = update_csv(rojo_no_overlap_df, f'{image_name}_rojo_No_overlap_Spots_statistics.csv')
        verde_overlap_df = update_csv(verde_overlap_df, f'{image_name}_verde_overlap_Spots_statistics.csv')
        verde_no_overlap_df = update_csv(verde_no_overlap_df, f'{image_name}_verde_No_overlap_Spots_statistics.csv')

        # Calculate average values for specified columns
        columns_to_average_rojo = ['Spot Intensity-Bg Subtract', 'Intensity-Bg Rojo']
        rojo_avg_overlap = calculate_average_values(rojo_overlap_df, 'TRACK_ID', columns_to_average_rojo)
        rojo_avg_no_overlap = calculate_average_values(rojo_no_overlap_df, 'TRACK_ID', columns_to_average_rojo)
        columns_to_average_verde = ['Spot Intensity-Bg Subtract', 'Intensity-Bg Verde']
        verde_avg_overlap = calculate_average_values(verde_overlap_df, 'TRACK_ID', columns_to_average_verde)
        verde_avg_no_overlap = calculate_average_values(verde_no_overlap_df, 'TRACK_ID', columns_to_average_verde)

        # Create directories for each image CSV
        image_directory = os.path.join(results_directory, image_name)
        os.makedirs(image_directory, exist_ok=True)
        csv_directory = os.path.join(image_directory, "csv")
        os.makedirs(csv_directory, exist_ok=True)

        # Create directories for each image PLOTS
        image_directory = os.path.join(results_directory, image_name)
        os.makedirs(image_directory, exist_ok=True)
        plots_directory = os.path.join(image_directory, "plots")
        os.makedirs(plots_directory, exist_ok=True)

        # Save updated dataframes to CSV files in the correct directory
        rojo_overlap_df.to_csv(os.path.join(csv_directory, f'{image_name}_rojo_overlap_Spots_statistics.csv'),
                               index=False)
        rojo_no_overlap_df.to_csv(os.path.join(csv_directory, f'{image_name}_rojo_No_overlap_Spots_statistics.csv'),
                                  index=False)
        verde_overlap_df.to_csv(os.path.join(csv_directory, f'{image_name}_verde_overlap_Spots_statistics.csv'),
                                index=False)
        verde_no_overlap_df.to_csv(os.path.join(csv_directory, f'{image_name}_verde_No_overlap_Spots_statistics.csv'),
                                   index=False)
        # Now call plot_overlapping_tracks after updating the CSV files
        plot_overlapping_tracks(image_name, rojo_overlap_df, verde_overlap_df, results_spots[image_name],
                                plots_directory)

# Process _Tracks statistics.csv files
for image_name in rojo_csv_files_tracks:
    if image_name in verde_csv_files_tracks:
        rojo_tracks_df = pd.read_csv(rojo_csv_files_tracks[image_name])
        verde_tracks_df = pd.read_csv(verde_csv_files_tracks[image_name])
        overlapping_rojo_ids = set(result['Rojo_TRACK_ID'] for result in results_spots[image_name])
        overlapping_verde_ids = set(result['Verde_TRACK_ID'] for result in results_spots[image_name])
        rojo_overlap_tracks_df = rojo_tracks_df[rojo_tracks_df['TRACK_ID'].isin(overlapping_rojo_ids)]
        rojo_no_overlap_tracks_df = rojo_tracks_df[~rojo_tracks_df['TRACK_ID'].isin(overlapping_rojo_ids)]
        verde_overlap_tracks_df = verde_tracks_df[verde_tracks_df['TRACK_ID'].isin(overlapping_verde_ids)]
        verde_no_overlap_tracks_df = verde_tracks_df[~verde_tracks_df['TRACK_ID'].isin(overlapping_verde_ids)]

        # Merge average values into track statistics dataframes
        rojo_overlap_tracks_df = rojo_overlap_tracks_df.merge(rojo_avg_overlap, on='TRACK_ID', how='left')
        rojo_no_overlap_tracks_df = rojo_no_overlap_tracks_df.merge(rojo_avg_no_overlap, on='TRACK_ID', how='left')
        verde_overlap_tracks_df = verde_overlap_tracks_df.merge(verde_avg_overlap, on='TRACK_ID', how='left')
        verde_no_overlap_tracks_df = verde_no_overlap_tracks_df.merge(verde_avg_no_overlap, on='TRACK_ID', how='left')

        # Create directories for each image
        image_directory = os.path.join(results_directory, image_name)
        os.makedirs(image_directory, exist_ok=True)
        csv_directory = os.path.join(image_directory, "csv")
        os.makedirs(csv_directory, exist_ok=True)

        # Save updated track statistics CSV files in the correct directory
        rojo_overlap_tracks_df.to_csv(os.path.join(csv_directory, f'{image_name}_rojo_overlap_Track_statistics.csv'),
                                      index=False)
        rojo_no_overlap_tracks_df.to_csv(
            os.path.join(csv_directory, f'{image_name}_rojo_No_overlap_Track_statistics.csv'), index=False)
        verde_overlap_tracks_df.to_csv(os.path.join(csv_directory, f'{image_name}_verde_overlap_Track_statistics.csv'),
                                       index=False)
        verde_no_overlap_tracks_df.to_csv(
            os.path.join(csv_directory, f'{image_name}_verde_No_overlap_Track_statistics.csv'), index=False)

        # Define file paths
        rojo_no_overlap_track_file = os.path.join(csv_directory, f'{image_name}_rojo_No_overlap_Track_statistics.csv')
        rojo_overlap_track_file = os.path.join(csv_directory, f'{image_name}_rojo_overlap_Track_statistics.csv')
        verde_no_overlap_track_file = os.path.join(csv_directory, f'{image_name}_verde_No_overlap_Track_statistics.csv')
        verde_overlap_track_file = os.path.join(csv_directory, f'{image_name}_verde_overlap_Track_statistics.csv')

# Extract specified column values and save to CSV
extract_column_values_single_file(column_indexes_no_overlap, column_indexes_overlap_rojo, column_indexes_overlap_verde)



def create_summary_csv(results_spots, rojo_dfs_spots, verde_dfs_spots, summary_analysis_directory):
    summary_data_rojo = []
    summary_data_verde = []

    for image_name, file_results in results_spots.items():
        results_file_path = os.path.join(results_directory, image_name, "csv", f'{image_name}_recalculated_trajectory_overlap_results.csv')
        results_df = pd.read_csv(results_file_path)

        total_tracks = len(results_df)
        n_overlapping_tracks = len(file_results)
        n_red_tracks_overlapping = len(set([result['Rojo_TRACK_ID'] for result in file_results]))
        n_green_tracks_overlapping = len(set([result['Verde_TRACK_ID'] for result in file_results]))

        rojo_classification_counts = {
            'Short Tracks': 0, 'Short Inmobile': 0, 'Short Confined': 0, 'Short Anomalous': 0,
            'Short Free': 0, 'Short Directed': 0, 'Long Confined': 0, 'Long Free': 0,
            'Long Directed': 0, 'Long Mobile': 0, 'Long Mobile Confined': 0, 'Long Mobile Free': 0,
            'Long Mobile Directed': 0, 'Immob': 0
        }
        verde_classification_counts = rojo_classification_counts.copy()

        for _, row in results_df.iterrows():
            if row['Track Length Rojo'] == 'Short':
                rojo_classification_counts['Short Tracks'] += 1
                if row['Motility Rojo'] == 'Inmobile':
                    rojo_classification_counts['Short Inmobile'] += 1
                rojo_classification_counts[f"Short {row['Alpha Movement Rojo']}"] += 1
            else:
                rojo_classification_counts[f"Long {row['sMSS Rojo Movement']}"] += 1
                if row['Motility Rojo'] == 'Mobile':
                    rojo_classification_counts['Long Mobile'] += 1
                    rojo_classification_counts[f"Long Mobile {row['sMSS Rojo Movement']}"] += 1
                elif row['Motility Rojo'] == 'Inmobile':
                    rojo_classification_counts['Immob'] += 1

            if row['Track Length Verde'] == 'Short':
                verde_classification_counts['Short Tracks'] += 1
                if row['Motility Verde'] == 'Inmobile':
                    verde_classification_counts['Short Inmobile'] += 1
                verde_classification_counts[f"Short {row['Alpha Movement Verde']}"] += 1
            else:
                verde_classification_counts[f"Long {row['sMSS Verde Movement']}"] += 1
                if row['Motility Verde'] == 'Mobile':
                    verde_classification_counts['Long Mobile'] += 1
                    verde_classification_counts[f"Long Mobile {row['sMSS Verde Movement']}"] += 1
                elif row['Motility Verde'] == 'Inmobile':
                    verde_classification_counts['Immob'] += 1

        summary_data_rojo.append({
            'Image Title': image_name,
            'Total Tracks': total_tracks,
            'N of Overlapping Tracks': n_overlapping_tracks,
            'N of Red Tracks overlapping': n_red_tracks_overlapping,
            **rojo_classification_counts
        })

        summary_data_verde.append({
            'Image Title': image_name,
            'Total Tracks': total_tracks,
            'N of Overlapping Tracks': n_overlapping_tracks,
            'N of Green Tracks overlapping': n_green_tracks_overlapping,
            **verde_classification_counts
        })

    # Crear DataFrames
    summary_df_rojo = pd.DataFrame(summary_data_rojo)
    summary_df_verde = pd.DataFrame(summary_data_verde)

    # Añadir fila 'Total' al final
    total_row_rojo = summary_df_rojo.drop(columns=['Image Title']).sum(numeric_only=True)
    total_row_rojo['Image Title'] = 'Total'
    summary_df_rojo = pd.concat([summary_df_rojo, pd.DataFrame([total_row_rojo])], ignore_index=True)

    total_row_verde = summary_df_verde.drop(columns=['Image Title']).sum(numeric_only=True)
    total_row_verde['Image Title'] = 'Total'
    summary_df_verde = pd.concat([summary_df_verde, pd.DataFrame([total_row_verde])], ignore_index=True)

    # Guardar los archivos
    summary_df_rojo.to_csv(os.path.join(summary_analysis_directory, 'summary_track_condition_rojo_overlapping.csv'),
                           index=False)
    summary_df_verde.to_csv(os.path.join(summary_analysis_directory, 'summary_track_condition_verde_overlapping.csv'),
                            index=False)


# Ejecutar al final del script
create_summary_csv(results_spots, rojo_dfs_spots, verde_dfs_spots, summary_analysis_directory)


