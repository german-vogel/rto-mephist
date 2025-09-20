import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
import pyperclip
import itertools
import pandas as pd
import json

# global parameters
K_rog_tor = 6.3e6
K_TF = 9.6e-3
K_rog_ind = 6.3e6
K_rog_PF1 = 6.23e6

# signal processing functions

def set_zero(sig):
    sig = np.asarray(sig)
    if sig.size < 200: return sig
    return sig - np.average(sig[10:200])

def integrate(time, sig):
    sig = np.asarray(sig)
    time = np.asarray(time)
    if len(sig) < 2: return np.zeros_like(sig)
    dt = (time[-1] - time[0]) / (len(time) - 1)
    return np.cumsum(sig) * dt

def preprocess(sig):
    sig = np.asarray(sig)
    if sig.size == 0: return sig
    d2y_max = 1
    sig2 = np.copy(sig)
    sig2[np.isnan(sig2)] = 0
    for i in range(len(sig2) - 1):
        if abs(sig2[i + 1] - sig2[i]) > d2y_max:
            sig2[i + 1] = sig2[i]
    return sig2

def RC_transform(time, sig, R, C):
    sig = np.asarray(sig)
    time = np.asarray(time)
    if len(sig) < 2: return np.zeros_like(sig)
    dt = (time[-1] - time[0]) / (len(time) - 1)
    out_sig = np.zeros_like(sig)
    for i in range(len(sig) - 1):
        out_sig[i + 1] = out_sig[i] + dt * ((sig[i] / R) - (out_sig[i] / (R * C)))
    return out_sig / C

def calculate_plasma_duration(time, Ip, threshold_amps):
    active_plasma_indices = np.where(Ip > threshold_amps)[0]
    if len(active_plasma_indices) == 0: return 0.0
    start_index, end_index = active_plasma_indices[0], active_plasma_indices[-1]
    return time[end_index] - time[start_index]

def process_shot_data(file_path, save_to_csv=False):
    try:
        with h5py.File(file_path, 'r') as f:
            Time_original = f['rogowski_coils']['rog_tor_coils']['time'][:] / 1000
            Time_internal_rog = f['rogowski_coils']['rog_internal']['time'][:] / 1000
            Time = np.linspace(0, 21e-3, 21000)

            U_rog_TF_raw = f['rogowski_coils']['rog_tor_coils']['data'][:]
            U_rog_CS_raw = f['rogowski_coils']['rog_inductor']['data'][:]
            U_rog_PF1_raw = f['rogowski_coils']['rog_pol_coils1']['data'][:]
            U_rog_PF2_raw = f['rogowski_coils']['rog_pol_coils2']['data'][:]
            U_rog_PF3_raw = f['rogowski_coils']['rog_pol_coils3']['data'][:]
            U_rog_int_raw = f['rogowski_coils']['rog_internal']['data'][:]

            U_rog_TF = np.interp(Time, Time_original, set_zero(U_rog_TF_raw))
            U_rog_CS = np.interp(Time, Time_original, set_zero(U_rog_CS_raw))
            U_rog_PF1 = np.interp(Time, Time_original, set_zero(U_rog_PF1_raw))
            U_rog_PF2 = np.interp(Time, Time_original, preprocess(U_rog_PF2_raw))
            U_rog_PF3 = np.interp(Time, Time_original, set_zero(U_rog_PF3_raw))
            U_rog_int = np.interp(Time, Time_internal_rog - 8.6e-5, U_rog_int_raw)

            I_TF = abs(integrate(Time, U_rog_TF)) * K_rog_tor
            B_phi = I_TF * K_TF

            Photod_raw = f['spectroscopy']['visible_emission']['data'][:]
            Photod_time_raw = f['spectroscopy']['visible_emission']['time'][:] / 1000
            Photod = np.interp(Time, Photod_time_raw, set_zero(Photod_raw))

            synt_sig = -RC_transform(Time, U_rog_TF, 1, 1.8e-4) * 0.665
            synt_sig -= 0.515 * (RC_transform(Time, U_rog_CS, 1, 8.6e-5) * 3 - RC_transform(Time, U_rog_CS, 1, 4.4e-4) * 1.67)
            synt_sig += 0.5 * 1.1 * (RC_transform(Time, U_rog_PF1, 1, 5.5e-4) * 3.8 + RC_transform(Time, U_rog_PF1, 1, 5e-5) * 0.65)
            synt_sig += RC_transform(Time, U_rog_PF2, 1, 4.5e-4) * 3.2 + RC_transform(Time, U_rog_PF2, 1, 15e-4) * 0.45
            synt_sig += 1.05 * (RC_transform(Time, U_rog_PF3, 1, 60e-4) * 0.4 + RC_transform(Time, U_rog_PF3, 1, 6e-4) * 2.4)

            U_rog_int_filt = U_rog_int - synt_sig
            Ip = integrate(Time, U_rog_int_filt) * 1.48e7 * 0.87
            Ip = Ip - Ip[6600]

            I_p_max_kA = np.max(Ip) / 1000.0
            duration_threshold_amps = 0.10 * np.max(Ip)
            plasma_duration_sec = calculate_plasma_duration(Time, Ip, duration_threshold_amps)
            B_phi_max = np.max(B_phi)

            wavelengths_Avantes, intensities_Avantes = np.array([]), np.array([])
            if 'spectroscopy' in f and 'Avantes' in f['spectroscopy']:
                try:
                    wavelengths_Avantes_raw = f['spectroscopy']['Avantes']['Wavelength'][:]
                    intensities_Avantes_raw = f['spectroscopy']['Avantes']['CleanI'][:]
                    if (wavelengths_Avantes_raw.size > 0 and intensities_Avantes_raw.size > 0 and
                        wavelengths_Avantes_raw.shape == intensities_Avantes_raw.shape):
                        max_intensity = np.max(intensities_Avantes_raw)
                        intensities_Avantes = intensities_Avantes_raw / max_intensity if max_intensity > 0 else np.zeros_like(intensities_Avantes_raw)
                        wavelengths_Avantes = wavelengths_Avantes_raw
                except:
                    pass  # silently ignore spectroscopy errors

            shot_number = os.path.basename(file_path).split('.')[0][:-2]

            # create dataframes for easy CSV export
            time_ms = Time * 1000
            main_data_df = pd.DataFrame({
                'time_ms': time_ms,
                'Bt_mT': B_phi,
                'Ip_kA': Ip / 1000,
                'H_alpha': Photod
            })
            
            spec_data_df = pd.DataFrame()
            if wavelengths_Avantes.size > 0 and intensities_Avantes.size > 0:
                spec_data_df = pd.DataFrame({
                    'wavelength_nm': wavelengths_Avantes,
                    'intensity': intensities_Avantes
                })

            # save to CSV
            if save_to_csv:
                local_folder = f"shot_{shot_number}"
                os.makedirs(local_folder, exist_ok=True)
                main_data_df.to_csv(f"{local_folder}/main_data.csv", index=False)
                if not spec_data_df.empty:
                    spec_data_df.to_csv(f"{local_folder}/spectroscopy.csv", index=False)
                
                # save metadata
                metadata = {
                    'shot_number': shot_number,
                    'I_p_max_kA': float(I_p_max_kA),
                    'B_phi_max_mT': float(B_phi_max),
                    'plasma_duration_ms': float(plasma_duration_sec * 1000)
                }
                with open(f"{local_folder}/metadata.json", "w") as f:
                    json.dump(metadata, f)

            return {
                'Time': Time, 'B_phi': B_phi, 'Ip': Ip, 'Photod': Photod,
                'wavelengths_Avantes': wavelengths_Avantes, 'intensities_Avantes': intensities_Avantes,
                'shot_number': shot_number, 'I_p_max_kA': I_p_max_kA, 'B_phi_max_mT': B_phi_max,
                'plasma_duration_sec': plasma_duration_sec,
                'main_data_df': main_data_df,
                'spec_data_df': spec_data_df
            }
    except Exception as e:
        messagebox.showerror("Data Load Error", f"Could not load or process file {file_path}:\n{e}")
        return None

class TokamakApp:
    def __init__(self, master):
        self.master = master
        master.title("MephiST-0 Tokamak Data Viewer")
        master.geometry("1200x1000")

        # matplotlib style configuration
        plt.rcParams.update({
            'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
            'ytick.labelsize': 8, 'legend.fontsize': 'small',
            'lines.linewidth': 1.2, 'axes.titlesize': 10,
            'figure.facecolor': 'white', 'axes.facecolor': 'white',
            'axes.edgecolor': 'black', 'axes.linewidth': 0.8,
            'grid.color': 'lightgray', 'grid.linestyle': '--',
            'grid.linewidth': 0.5
        })

        self.file_paths = []
        self.processed_data = []
        self.color_palette = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']

        # variables for cursor dynamics
        self.cursor_dynamics_enabled = False
        self.cursor_lines = []
        self.residual_cursor_lines = []
        self.last_cursor_x = None
        self.motion_cid = None
        self.right_click_cid = None
        
        # Variables for zoom synchronization
        self.syncing_zoom = False
        self.time_axes = []  # Will store references to time-based axes
        self.residual_axes = []  # Will store references to residual axes

        self.create_widgets()

    def create_widgets(self):
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
    
        self.top_button_frame = tk.Frame(self.main_frame)
        self.top_button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(self.top_button_frame, text="Load Shot", command=self.load_shot).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.top_button_frame, text="Clear Shots", command=self.clear_shots).pack(side=tk.LEFT, padx=5, pady=5)
        self.cursor_toggle_button = tk.Button(self.top_button_frame, text="Enable Cursor Dynamics", command=self.toggle_cursor_dynamics)
        self.cursor_toggle_button.pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.top_button_frame, text="Save Data to CSV", command=self.save_data_to_csv).pack(side=tk.LEFT, padx=5, pady=5)

        self.plot_frame = tk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_container = tk.Frame(self.plot_frame)
        self.canvas_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # create 4x2 figure for the 4 main panels and 4 residual panels
        self.fig = plt.figure(figsize=(12, 12), facecolor='white')
        gs = self.fig.add_gridspec(4, 2, height_ratios=[3, 1, 3, 1], hspace=0.4)
    
        # configure main axes and residual axes
        self.ax_bt = self.fig.add_subplot(gs[0, 0])
        self.ax_bt_residual = self.fig.add_subplot(gs[1, 0], sharex=self.ax_bt)
        
        self.ax_ip = self.fig.add_subplot(gs[0, 1])
        self.ax_ip_residual = self.fig.add_subplot(gs[1, 1], sharex=self.ax_ip)
        
        self.ax_halpha = self.fig.add_subplot(gs[2, 0])
        self.ax_halpha_residual = self.fig.add_subplot(gs[3, 0], sharex=self.ax_halpha)
        
        self.ax_avantes = self.fig.add_subplot(gs[2, 1])
        self.ax_avantes_residual = self.fig.add_subplot(gs[3, 1], sharex=self.ax_avantes)
        
        # Store time-based axes for synchronization
        self.time_axes = [self.ax_bt, self.ax_ip, self.ax_halpha]
        self.time_residual_axes = [self.ax_bt_residual, self.ax_ip_residual, self.ax_halpha_residual]
        self.spec_axes = [self.ax_avantes]
        self.spec_residual_axes = [self.ax_avantes_residual]
    
        # white background for all axes
        for ax in [self.ax_bt, self.ax_ip, self.ax_halpha, self.ax_avantes,
                  self.ax_bt_residual, self.ax_ip_residual, self.ax_halpha_residual, self.ax_avantes_residual]:
            ax.set_facecolor('white')
    
        # set permanent axis labels
        self.ax_bt.set_ylabel('Bt [mT]')
        self.ax_bt_residual.set_ylabel('ΔBt [mT]')
        self.ax_bt_residual.set_xlabel('Time [ms]')
        
        self.ax_ip.set_ylabel('Ip [kA]')
        self.ax_ip_residual.set_ylabel('ΔIp [kA]')
        self.ax_ip_residual.set_xlabel('Time [ms]')
        
        self.ax_halpha.set_ylabel('H-alpha [a.u.]')
        self.ax_halpha_residual.set_ylabel('ΔH-alpha [a.u.]')
        self.ax_halpha_residual.set_xlabel('Time [ms]')
        
        self.ax_avantes.set_ylabel('Intensity [a.u.]')
        self.ax_avantes_residual.set_ylabel('ΔIntensity [a.u.]')
        self.ax_avantes_residual.set_xlabel('Wavelength [nm]')
        
        # Hide x labels for main panels (residual panels will show them)
        for ax in [self.ax_bt, self.ax_ip, self.ax_halpha, self.ax_avantes]:
            ax.tick_params(labelbottom=False)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_container)
        canvas_widget = self.canvas.get_tk_widget()

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_container)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.data_box_label = tk.Label(self.toolbar, text="", anchor="w", justify="left", font=("Courier New", 8))
        self.data_box_label.pack(side=tk.LEFT, padx=10)

        # Connect zoom/pan events for synchronization
        self.connect_zoom_events()
        
        self.canvas.draw()

    def connect_zoom_events(self):
        """Connect zoom/pan events for all time-based axes"""
        for ax in self.time_axes:
            ax.callbacks.connect('xlim_changed', self.sync_time_axes)
        for ax in self.spec_axes:
            ax.callbacks.connect('xlim_changed', self.sync_spec_axes)

    def sync_time_axes(self, event_ax):
        """Synchronize all time-based axes when one is zoomed/panned"""
        if self.syncing_zoom:
            return
            
        self.syncing_zoom = True
        
        try:
            # Get current x-limits from the axis that triggered the event
            xmin, xmax = event_ax.get_xlim()
            
            # Apply the same limits to all time-based axes
            for ax in self.time_axes:
                if ax != event_ax:
                    ax.set_xlim(xmin, xmax)
                    
            # Adjust y-axis limits for all time-based axes
            for ax in self.time_axes + self.time_residual_axes:
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
                
            self.canvas.draw_idle()
            
        finally:
            self.syncing_zoom = False
            
    def sync_spec_axes(self, event_ax):
        """Synchronize spectroscopy axes when one is zoomed/panned"""
        if self.syncing_zoom:
            return
            
        self.syncing_zoom = True
        
        try:
            # Get current x-limits from the axis that triggered the event
            xmin, xmax = event_ax.get_xlim()
            
            # Apply the same limits to all spectroscopy axes
            for ax in self.spec_axes:
                if ax != event_ax:
                    ax.set_xlim(xmin, xmax)
                    
            # Adjust y-axis limits for all spectroscopy axes
            for ax in self.spec_axes + self.spec_residual_axes:
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
                
            self.canvas.draw_idle()
            
        finally:
            self.syncing_zoom = False

    def load_shot(self):
        file_path = filedialog.askopenfilename(
            title="Select MephiST-0 Shot File",
            filetypes=[("NXS files", "*.nxs"), ("HDF5 files", "*.hdf5 *.h5"), ("All files", "*.*")]
        )
        if not file_path:
            return
        
        try:
            shot_data = process_shot_data(file_path, save_to_csv=False)
            if shot_data:
                shot_number = shot_data['shot_number']
                
                # check if shot already loaded
                for data in self.processed_data:
                    if data and data['shot_number'] == shot_number:
                        messagebox.showinfo("Shot Already Loaded", f"Shot {shot_number} is already loaded.")
                        return
                
                self.file_paths.append(file_path)
                self.processed_data.append(shot_data)
                self.plot_data()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load shot: {e}")

    def save_data_to_csv(self):
        if not self.processed_data:
            messagebox.showinfo("No Data", "No shot data to save.")
            return
            
        for data in self.processed_data:
            shot_number = data['shot_number']
            process_shot_data(self.file_paths[self.processed_data.index(data)], save_to_csv=True)
            
        messagebox.showinfo("Success", f"Data saved for {len(self.processed_data)} shot(s) in their respective folders.")

    def clear_shots(self):
        self.file_paths = []
        self.processed_data = []
        self.plot_data()

    def toggle_cursor_dynamics(self):
        self.cursor_dynamics_enabled = not self.cursor_dynamics_enabled
        self.cursor_toggle_button.config(
            text="Disable Cursor Dynamics" if self.cursor_dynamics_enabled else "Enable Cursor Dynamics"
        )
        
        if self.cursor_dynamics_enabled:
            self.connect_cursor_events()
        else:
            self.disconnect_cursor_events()
            self.clear_cursor_lines()
            self.data_box_label.config(text="")
        
        self.canvas.draw()

    def connect_cursor_events(self):
        self.motion_cid = self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.right_click_cid = self.canvas.mpl_connect('button_press_event', self.on_right_click)

    def disconnect_cursor_events(self):
        if hasattr(self, 'motion_cid'):
            self.canvas.mpl_disconnect(self.motion_cid)
            del self.motion_cid
        if hasattr(self, 'right_click_cid'):
            self.canvas.mpl_disconnect(self.right_click_cid)
            del self.right_click_cid

    def on_mouse_move(self, event):
        if not event.inaxes or not self.cursor_dynamics_enabled:
            return

        x = event.xdata
        self.last_cursor_x = x 

        self.clear_cursor_lines()

        # Draw vertical line in appropriate axes based on which panel the cursor is in
        if event.inaxes in self.time_axes + self.time_residual_axes:
            # Time-based panels
            for ax in self.time_axes + self.time_residual_axes:
                self.cursor_lines.append(ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.8))
        elif event.inaxes in self.spec_axes + self.spec_residual_axes:
            # Spectroscopy panels
            for ax in self.spec_axes + self.spec_residual_axes:
                self.cursor_lines.append(ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.8))

        # update data in label
        header = "Shot\tTime(ms)\tBt(mT)\tIp(kA)\tH-alpha(a.u.)"
        data_table = [header]
        
        for data in self.processed_data:
            if data is None:
                continue
                
            time_ms = data['Time'] * 1000
            idx = (np.abs(time_ms - x)).argmin()
            
            shot_num = data['shot_number']
            t = time_ms[idx]
            bt = data['B_phi'][idx]
            ip = data['Ip'][idx] / 1000
            ha = data['Photod'][idx]

            row = f"{shot_num}\t{t:.2f}\t{bt:.2f}\t{ip:.2f}\t{ha:.3f}"
            data_table.append(row)

        self.data_box_label.config(text="\n".join(data_table))
        self.canvas.draw_idle()

    def on_right_click(self, event):
        if not event.inaxes or not self.cursor_dynamics_enabled or event.button != 3:
            return

        x = event.xdata
        clipboard_text = "Shot\tTime(ms)\tBt(mT)\tIp(kA)\tH-alpha(a.u.)\n"
        
        for data in self.processed_data:
            if data is None:
                continue
                
            time_ms = data['Time'] * 1000
            idx = (np.abs(time_ms - x)).argmin()
            
            shot_num = data['shot_number']
            t = time_ms[idx]
            bt = data['B_phi'][idx]
            ip = data['Ip'][idx] / 1000
            ha = data['Photod'][idx]

            row = f"{shot_num}\t{t:.2f}\t{bt:.2f}\t{ip:.2f}\t{ha:.3f}\n"
            clipboard_text += row

        pyperclip.copy(clipboard_text)
        messagebox.showinfo("Copied", "Cursor data copied to clipboard.")

    def clear_cursor_lines(self):
        for line in self.cursor_lines:
            try:
                line.remove()
            except:
                pass
        self.cursor_lines.clear()

    def plot_data(self):
        # clear plot data but keep axis labels
        all_axes = [self.ax_bt, self.ax_ip, self.ax_halpha, self.ax_avantes,
                   self.ax_bt_residual, self.ax_ip_residual, self.ax_halpha_residual, self.ax_avantes_residual]
        
        for ax in all_axes:
            ax.clear()
            ax.set_facecolor('white')
            
        # reapply permanent axis labels
        self.ax_bt.set_ylabel('Bt [mT]')
        self.ax_bt_residual.set_ylabel('ΔBt [mT]')
        self.ax_bt_residual.set_xlabel('Time [ms]')
        
        self.ax_ip.set_ylabel('Ip [kA]')
        self.ax_ip_residual.set_ylabel('ΔIp [kA]')
        self.ax_ip_residual.set_xlabel('Time [ms]')
        
        self.ax_halpha.set_ylabel('H-alpha [a.u.]')
        self.ax_halpha_residual.set_ylabel('ΔH-alpha [a.u.]')
        self.ax_halpha_residual.set_xlabel('Time [ms]')
        
        self.ax_avantes.set_ylabel('Intensity [a.u.]')
        self.ax_avantes_residual.set_ylabel('ΔIntensity [a.u.]')
        self.ax_avantes_residual.set_xlabel('Wavelength [nm]')
        
        # Hide x labels for main panels
        for ax in [self.ax_bt, self.ax_ip, self.ax_halpha, self.ax_avantes]:
            ax.tick_params(labelbottom=False)

        if not self.processed_data:
            self.canvas.draw()
            return

        color_cycle = itertools.cycle(self.color_palette)
        colors = []
        
        for data in self.processed_data:
            color = next(color_cycle)
            colors.append(color)
            shot_label = data['shot_number']
            time_ms = data['Time'] * 1000
            
            # plot time data
            self.ax_bt.plot(time_ms, data['B_phi'], label=shot_label, color=color)
            self.ax_ip.plot(time_ms, data['Ip'] / 1000, label=shot_label, color=color)
            self.ax_halpha.plot(time_ms, data['Photod'], label=shot_label, color=color)
            
            # plot spectrum (if available and valid)
            if (data['wavelengths_Avantes'].size > 0 and 
                data['intensities_Avantes'].size > 0 and
                data['wavelengths_Avantes'].shape == data['intensities_Avantes'].shape):
                self.ax_avantes.plot(data['wavelengths_Avantes'], data['intensities_Avantes'], 
                                    label=shot_label, color=color)

        # Plot residuals if we have exactly 2 shots
        if len(self.processed_data) == 2:
            data1, data2 = self.processed_data
            time_ms1 = data1['Time'] * 1000
            time_ms2 = data2['Time'] * 1000
            
            # Ensure both time arrays are the same length for residual calculation
            if len(time_ms1) == len(time_ms2):
                # Bt residual
                bt_residual = data1['B_phi'] - data2['B_phi']
                self.ax_bt_residual.plot(time_ms1, bt_residual, color='red', label='Difference')
                self.ax_bt_residual.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
                
                # Ip residual
                ip_residual = (data1['Ip'] - data2['Ip']) / 1000
                self.ax_ip_residual.plot(time_ms1, ip_residual, color='red', label='Difference')
                self.ax_ip_residual.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
                
                # H-alpha residual
                ha_residual = data1['Photod'] - data2['Photod']
                self.ax_halpha_residual.plot(time_ms1, ha_residual, color='red', label='Difference')
                self.ax_halpha_residual.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            
            # Spectroscopy residual
            if (data1['wavelengths_Avantes'].size > 0 and data2['wavelengths_Avantes'].size > 0 and
                np.array_equal(data1['wavelengths_Avantes'], data2['wavelengths_Avantes'])):
                spec_residual = data1['intensities_Avantes'] - data2['intensities_Avantes']
                self.ax_avantes_residual.plot(data1['wavelengths_Avantes'], spec_residual, color='red', label='Difference')
                self.ax_avantes_residual.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        # configure grid and limits
        for ax in all_axes:
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        if self.ax_avantes.has_data():
            self.ax_avantes.set_xlim(350, 1000)
            self.ax_avantes.set_ylim(0, 1.1)
            
            if len(self.processed_data) == 2 and self.ax_avantes_residual.has_data():
                self.ax_avantes_residual.set_xlim(350, 1000)
        else:
            self.ax_avantes.text(0.5, 0.5, 'No spectroscopy data', 
                               horizontalalignment='center', verticalalignment='center',
                               transform=self.ax_avantes.transAxes, fontsize=10, alpha=0.7)
            self.ax_avantes_residual.text(0.5, 0.5, 'No spectroscopy data', 
                                        horizontalalignment='center', verticalalignment='center',
                                        transform=self.ax_avantes_residual.transAxes, fontsize=10, alpha=0.7)

        # add legends
        for ax in [self.ax_bt, self.ax_ip, self.ax_halpha, self.ax_avantes]:
            if ax.has_data():
                ax.legend(loc='best', fontsize='small')
                
        for ax in [self.ax_bt_residual, self.ax_ip_residual, self.ax_halpha_residual, self.ax_avantes_residual]:
            if ax.has_data():
                ax.legend(loc='best', fontsize='small')

        # Reconnect zoom events after plotting new data
        self.connect_zoom_events()
        
        # redraw cursor if active
        if self.cursor_dynamics_enabled:
            self.draw_cursor_at(self.last_cursor_x)
            
        self.canvas.draw()

    def draw_cursor_at(self, x):
        """Redraw cursor at specific position"""
        if x is None or not self.cursor_dynamics_enabled:
            return
            
        self.clear_cursor_lines()
        
        # Draw cursor in all time-based panels
        for ax in self.time_axes + self.time_residual_axes:
            self.cursor_lines.append(ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.8))
            
        # update data
        header = "Shot\tTime(ms)\tBt(mT)\tIp(kA)\tH-alpha(a.u.)"
        data_table = [header]
        
        for data in self.processed_data:
            if data is None:
                continue
                
            time_ms = data['Time'] * 1000
            idx = (np.abs(time_ms - x)).argmin()
            
            shot_num = data['shot_number']
            t = time_ms[idx]
            bt = data['B_phi'][idx]
            ip = data['Ip'][idx] / 1000
            ha = data['Photod'][idx]

            row = f"{shot_num}\t{t:.2f}\t{bt:.2f}\t{ip:.2f}\t{ha:.3f}"
            data_table.append(row)

        self.data_box_label.config(text="\n".join(data_table))
        self.canvas.draw_idle()

    @staticmethod
    def lighter_color(color, factor=1.5):
        r, g, b = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        return f"#{r:02x}{g:02x}{b:02x}"

if __name__ == "__main__":
    # Windows DPI awareness fix
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except:
            pass
    
    root = tk.Tk()
    root.tk.call('tk', 'scaling', 2.0)
    
    app = TokamakApp(root)
    root.mainloop()