import argparse
import numpy as np
import os
import sys
import json
import platform
import matplotlib


def _use_working_backend():
    """Pick a GUI backend that actually renders on this machine.

    macOS ships a deprecated, buggy system Tcl/Tk (8.5) that frequently
    fails to show or focus TkAgg windows, so the native Cocoa backend is
    tried first there. Windows/Linux Tk installs are reliable, so TkAgg
    stays first everywhere else. Each candidate is validated by actually
    creating a figure, since matplotlib.use() alone doesn't catch missing
    dependencies (e.g. Qt bindings not installed).
    """
    candidates = (
        ["MacOSX", "TkAgg", "QtAgg", "Qt5Agg"]
        if platform.system() == "Darwin"
        else ["TkAgg", "QtAgg", "Qt5Agg"]
    )
    for name in candidates:
        try:
            matplotlib.use(name, force=True)
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.close(fig)
            return name
        except Exception:
            continue
    return matplotlib.get_backend()


_BACKEND = _use_working_backend()
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

print(f"  (backend gráfico: {_BACKEND})")


def main(path: str, t0: float = None, win_len: float = None):
    data = np.load(path, allow_pickle=True)
    peaks = data["peaks"].tolist()
    troughs = data["troughs"].tolist()
    times = data["times"]
    signal = data["rsp_clean"]
    fs = float(data["fs"])
    seg_name = str(data.get("seg", "SEG"))
    
    # Track changes in this session
    initial_peaks = set(peaks)
    initial_troughs = set(troughs)
    changes = {
        "peaks_added": [],
        "peaks_removed": [],
        "troughs_added": [],
        "troughs_removed": []
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, signal, lw=1, color="#2ca02c", label="Señal filtrada")
    scat_peaks = ax.scatter(np.array(peaks)/fs, signal[peaks], c="red", marker="^", label="Peaks")
    scat_troughs = ax.scatter(np.array(troughs)/fs, signal[troughs], c="orange", marker="v", label="Troughs")
    ax.set_title(f"Editor interactivo — {seg_name}\nSelecciona modo y haz clic sobre la señal", fontsize=10)
    ax.legend(loc="upper right")
    ax.set_xlabel("Tiempo (s)")
    ax.grid(True)

    def refresh():
        scat_peaks.set_offsets(np.column_stack((np.array(peaks)/fs, signal[peaks])))
        scat_troughs.set_offsets(np.column_stack((np.array(troughs)/fs, signal[troughs])))
        fig.canvas.draw_idle()

    # Mode: 'add_peak', 'add_trough', 'remove_peak', 'remove_trough'
    current_mode = 'remove_trough'
    
    def set_mode_add_peak(event):
        nonlocal current_mode
        current_mode = 'add_peak'
        btn_add_peak.color = 'lightgreen'
        btn_add_trough.color = 'lightgray'
        btn_remove_peak.color = 'lightgray'
        btn_remove_trough.color = 'lightgray'
        fig.canvas.draw_idle()
        print("  Modo: Añadir Peak (clic)")
    
    def set_mode_add_trough(event):
        nonlocal current_mode
        current_mode = 'add_trough'
        btn_add_peak.color = 'lightgray'
        btn_add_trough.color = 'lightgreen'
        btn_remove_peak.color = 'lightgray'
        btn_remove_trough.color = 'lightgray'
        fig.canvas.draw_idle()
        print("  Modo: Añadir Trough (clic)")
    
    def set_mode_remove_peak(event):
        nonlocal current_mode
        current_mode = 'remove_peak'
        btn_add_peak.color = 'lightgray'
        btn_add_trough.color = 'lightgray'
        btn_remove_peak.color = 'lightcoral'
        btn_remove_trough.color = 'lightgray'
        fig.canvas.draw_idle()
        print("  Modo: Eliminar Peak (clic)")
    
    def set_mode_remove_trough(event):
        nonlocal current_mode
        current_mode = 'remove_trough'
        btn_add_peak.color = 'lightgray'
        btn_add_trough.color = 'lightgray'
        btn_remove_peak.color = 'lightgray'
        btn_remove_trough.color = 'lightcoral'
        fig.canvas.draw_idle()
        print("  Modo: Eliminar Trough (clic)")

    def on_click(event):
        # Ignore clicks outside the main plot (e.g. on the mode/Save buttons)
        if event.inaxes != ax:
            return
        if event.xdata is None:
            return

        # Ignore clicks while the matplotlib pan/zoom tools are active, so
        # navigation still works (no modifier key needed to edit — a plain
        # left-click in the selected mode performs the action. This is what
        # makes it work on macOS, where Ctrl+click is hijacked as right-click).
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return

        # Only act on the primary (left) button
        if event.button not in (1, None):
            return

        idx = int(event.xdata * fs)
        idx = np.clip(idx, 0, len(signal)-1)
        
        if current_mode == 'add_peak':
            peaks.append(idx)
            t_added = idx / fs
            changes["peaks_added"].append(t_added)
            print(f"  ✓ Añadido peak en {t_added:.2f}s")
            refresh()
        elif current_mode == 'add_trough':
            troughs.append(idx)
            t_added = idx / fs
            changes["troughs_added"].append(t_added)
            print(f"  ✓ Añadido trough en {t_added:.2f}s")
            refresh()
        elif current_mode == 'remove_peak':
            if not peaks:
                print("  ⚠ No hay peaks para eliminar")
                return
            d_peaks = np.abs(np.array(peaks) - idx)
            remove_idx = int(np.argmin(d_peaks))
            removed_time = peaks[remove_idx] / fs
            peaks.pop(remove_idx)
            changes["peaks_removed"].append(removed_time)
            print(f"  ✓ Eliminado peak en {removed_time:.2f}s")
            refresh()
        elif current_mode == 'remove_trough':
            if not troughs:
                print("  ⚠ No hay troughs para eliminar")
                return
            d_troughs = np.abs(np.array(troughs) - idx)
            remove_idx = int(np.argmin(d_troughs))
            removed_time = troughs[remove_idx] / fs
            troughs.pop(remove_idx)
            changes["troughs_removed"].append(removed_time)
            print(f"  ✓ Eliminado trough en {removed_time:.2f}s")
            refresh()

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
    modified = False

    def guardar(event):
        nonlocal modified
        modified = True
        print(f"\n  ✓ Guardando cambios: {len(peaks)} peaks, {len(troughs)} troughs")
        np.savez(
            path,
            peaks=np.array(sorted(set(peaks))),
            troughs=np.array(sorted(set(troughs))),
            times=times,
            rsp_clean=signal,
            fs=fs,
            seg=seg_name,
        )
        
        # Save changes to a separate JSON file
        changes_file = path + ".changes.json"
        with open(changes_file, 'w') as f:
            json.dump(changes, f)
        
        # Create a status file to indicate save was successful
        status_file = path + ".saved"
        with open(status_file, 'w') as f:
            f.write("saved")
        fig.canvas.mpl_disconnect(cid_click)
        plt.close(fig)

    def cancelar(event):
        print(f"\n  ✗ Cancelado - no se guardaron cambios")
        # Create a status file to indicate cancel
        status_file = path + ".saved"
        with open(status_file, 'w') as f:
            f.write("cancelled")
        fig.canvas.mpl_disconnect(cid_click)
        plt.close(fig)

    # Create mode buttons
    ax_add_peak = plt.axes([0.02, 0.02, 0.12, 0.04])
    ax_add_trough = plt.axes([0.15, 0.02, 0.12, 0.04])
    ax_remove_peak = plt.axes([0.28, 0.02, 0.12, 0.04])
    ax_remove_trough = plt.axes([0.41, 0.02, 0.12, 0.04])
    ax_guardar = plt.axes([0.75, 0.02, 0.1, 0.04])
    ax_cancelar = plt.axes([0.86, 0.02, 0.1, 0.04])
    
    btn_add_peak = Button(ax_add_peak, "Añadir Peak")
    btn_add_trough = Button(ax_add_trough, "Añadir Trough")
    btn_remove_peak = Button(ax_remove_peak, "Eliminar Peak")
    btn_remove_trough = Button(ax_remove_trough, "Eliminar Trough")
    
    btn_add_peak.on_clicked(set_mode_add_peak)
    btn_add_trough.on_clicked(set_mode_add_trough)
    btn_remove_peak.on_clicked(set_mode_remove_peak)
    btn_remove_trough.on_clicked(set_mode_remove_trough)
    
    # Set initial mode
    btn_remove_trough.color = 'lightcoral'
    
    # Guardar y Cancelar - guardar referencia para que no se pierda
    btn_guardar = Button(ax_guardar, "Guardar")
    btn_cancelar = Button(ax_cancelar, "Cancelar")
    btn_guardar.on_clicked(guardar)
    btn_cancelar.on_clicked(cancelar)
    
    if t0 is not None and win_len is not None:
        ax.set_xlim(t0, t0 + win_len)
    fig.subplots_adjust(bottom=0.15, top=0.95)
    
    plt.show(block=True)
    
    # If window closed without buttons
    status_file = path + ".saved"
    if not os.path.exists(status_file):
        print("  ⚠ Ventana cerrada sin guardar")
        with open(status_file, 'w') as f:
            f.write("cancelled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    parser.add_argument("--t0", type=float, default=None)
    parser.add_argument("--win_len", type=float, default=None)
    args = parser.parse_args()
    main(args.filepath, t0=args.t0, win_len=args.win_len)
