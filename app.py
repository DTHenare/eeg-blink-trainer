import streamlit as st
import mne
import numpy as np
import matplotlib.pyplot as plt
 
# --- 1. App Configuration ---
st.set_page_config(page_title="Advanced EEG Artifact Trainer", layout="wide")
 
st.title("👁️ Advanced EEG Artifact Trainer")
st.markdown("""
**Goal:** Distinguish between **Blinks (Vertical)**, **Saccades (Horizontal)**, and **Clean Data**.
""")
 
# --- 2. Data Loading & Processing ---
@st.cache_resource
def load_data():
    sample_data_folder = mne.datasets.sample.data_path()
    raw_fname = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    
    # Load data
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.pick_types(meg=False, eeg=True, eog=True)
    raw.filter(1, 40, verbose=False)
    
    # --- FIX: Create Virtual HEOG Channel ---
    # The dataset lacks 'EOG 062'. We create it by subtracting Left (EEG 005) - Right (EEG 010).
    # This creates a bipolar channel sensitive to horizontal eye movements.
    
    # 1. Get data from two lateral EEG channels (approx F7 and F8 positions)
    left_eeg = raw.get_data(picks=['EEG 005'])[0]
    right_eeg = raw.get_data(picks=['EEG 010'])[0]
    
    # 2. Subtract them to isolate horizontal movement (Common mode noise is cancelled)
    heog_data = left_eeg - right_eeg
    
    # 3. Create a new Info object for this virtual channel
    info = mne.create_info(['EOG 062'], raw.info['sfreq'], ['eog'])
    virtual_heog_raw = mne.io.RawArray(heog_data.reshape(1, -1), info)
    
    # 4. Add it back to the main raw object
    raw.add_channels([virtual_heog_raw], force_update_info=True)
    
    return raw
 
# Initialize Session State
if 'raw' not in st.session_state:
    with st.spinner('Loading and analyzing EEG Data... (Generating Virtual HEOG)'):
        raw = load_data()
        st.session_state.raw = raw
        
        # --- Ground Truth Generation ---
        # 1. Detect Blinks (Vertical EOG 061)
        eog_events_v = mne.preprocessing.find_eog_events(raw, ch_name='EOG 061', verbose=False)
        st.session_state.blink_events = eog_events_v
 
        # 2. Detect Horizontal Movements (Virtual EOG 062)
        # We use a lower threshold for HEOG as the signal is often smaller
        eog_events_h = mne.preprocessing.find_eog_events(raw, ch_name='EOG 062', thresh=40e-6, verbose=False)
        st.session_state.horiz_events = eog_events_h
 
# --- 3. Game State Defaults ---
if 'current_start_time' not in st.session_state:
    st.session_state.current_start_time = 10.0
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'attempts' not in st.session_state:
    st.session_state.attempts = 0
if 'show_answer' not in st.session_state:
    st.session_state.show_answer = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = ""
 
# --- 4. Logic Helper Functions ---
def check_artifacts(start_time, duration=3.0):
    sfreq = st.session_state.raw.info['sfreq']
    
    # Helper to check if events exist in window
    def has_event(events):
        if events is None or len(events) == 0: return False
        ev_times = events[:, 0] / sfreq
        return np.any((ev_times >= start_time) & (ev_times <= start_time + duration))
 
    has_blink = has_event(st.session_state.blink_events)
    has_horiz = has_event(st.session_state.horiz_events)
    
    # Determine Ground Truth Label
    if has_blink:
        return "Blink"
    elif has_horiz:
        return "Horizontal Move"
    else:
        return "Clean / No Artifact"
 
def next_segment():
    raw = st.session_state.raw
    max_time = raw.times[-1] - 5.0
    
    # 50% chance to jump to a known artifact, 50% random
    if np.random.rand() > 0.5:
        all_events = np.vstack([st.session_state.blink_events, st.session_state.horiz_events])
        random_event_sample = all_events[np.random.randint(len(all_events)), 0]
        new_time = (random_event_sample / raw.info['sfreq']) - 1.0
        new_time = np.clip(new_time, 0, max_time)
        st.session_state.current_start_time = new_time
    else:
        st.session_state.current_start_time = np.random.uniform(0, max_time)
        
    st.session_state.show_answer = False
 
# --- 5. UI Layout ---
col1, col2 = st.columns([3, 1])
 
with col1:
    st.subheader("Raw EEG Trace")
    
    duration = 3.0
    start = st.session_state.current_start_time
    raw = st.session_state.raw
    
    t_idx_start = int(start * raw.info['sfreq'])
    t_idx_stop = int((start + duration) * raw.info['sfreq'])
    times = raw.times[t_idx_start:t_idx_stop]
    
    # Visualization Channels
    # 001/002 = Frontal (Blinks)
    # 005/010 = Lateral (Horizontal)
    eeg_picks = ['EEG 001', 'EEG 002', 'EEG 005', 'EEG 010']
    data_eeg, _ = raw[eeg_picks, t_idx_start:t_idx_stop]
    
    # Get Answer Key Data
    data_veog, _ = raw['EOG 061', t_idx_start:t_idx_stop] # Vertical
    data_heog, _ = raw['EOG 062', t_idx_start:t_idx_stop] # Virtual Horizontal
 
    # --- PLOTTING ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 1. EEG Plot
    for i, ch in enumerate(eeg_picks):
        ax1.plot(times, data_eeg[i].T + (i * 0.00015), label=ch)
    ax1.set_title("EEG Channels (Frontal & Lateral)")
    ax1.set_yticks([])
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # 2. Answer Key: Vertical EOG
    if st.session_state.show_answer:
        ax2.plot(times, data_veog[0].T, color='red')
        ax2.set_title("Vertical EOG (Blink Detector)")
    else:
        ax2.text(times[len(times)//2], 0, "Hidden", ha='center', color='gray')
        ax2.set_title("Vertical EOG (Hidden)")
    ax2.set_yticks([])
 
    # 3. Answer Key: Horizontal EOG
    if st.session_state.show_answer:
        ax3.plot(times, data_heog[0].T, color='blue')
        ax3.set_title("Horizontal EOG (Computed Saccade Detector)")
    else:
        ax3.text(times[len(times)//2], 0, "Hidden", ha='center', color='gray')
        ax3.set_title("Horizontal EOG (Hidden)")
    ax3.set_yticks([])
    ax3.set_xlabel("Time (s)")
    
    st.pyplot(fig)
 
with col2:
    st.markdown("### Diagnosis")
    st.write(f"Window Start: **{start:.2f}s**")
    
    with st.form("diagnosis_form"):
        options = ["Clean / No Artifact", "Blink", "Horizontal Move"]
        user_choice = st.radio("What is dominant?", options)
        submit_btn = st.form_submit_button("Submit")
    
    if submit_btn and not st.session_state.show_answer:
        st.session_state.attempts += 1
        truth = check_artifacts(start, duration)
        
        if user_choice == truth:
            st.success(f"Correct! It was {truth}.")
            st.session_state.score += 1
        else:
            st.error(f"Incorrect. It was actually: {truth}")
            
        st.session_state.show_answer = True
        st.rerun()
 
    if st.session_state.show_answer:
        if st.button("Next Segment ➡️"):
            next_segment()
            st.rerun()
 
    st.markdown("---")
    st.metric("Score", f"{st.session_state.score} / {st.session_state.attempts}")
 
# --- 6. Educational Sidebar ---
st.sidebar.title("Artifact Guide")
st.sidebar.markdown("""
### 1. Blinks (Vertical)
* **Shape:** Large 'V' or 'U' dip.
* **Channels:** Strongest in **EEG 001/002**.
* **Truth:** Matches Red Graph.
 
### 2. Saccades (Horizontal)
* **Shape:** Boxy / Steps.
* **Channels:** **EEG 005 and 010** move in *opposite* directions (one goes up, one goes down).
* **Truth:** Matches Blue Graph.
""")
