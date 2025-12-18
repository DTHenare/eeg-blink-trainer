import streamlit as st
import mne
import numpy as np
import matplotlib.pyplot as plt
 
# --- 1. App Configuration & Setup ---
st.set_page_config(page_title="EEG Blink Trainer", layout="wide")
 
st.title("🧠 EEG Blink Artifact Trainer")
st.markdown("""
**Goal:** Learn to identify eye-blink artifacts in raw EEG data.
Blinks typically appear as **high-amplitude, low-frequency deflections**, most prominent in frontal channels (e.g., Fp1, Fp2).
""")
 
# --- 2. Data Loading (Cached for Speed) ---
@st.cache_resource
def load_data():
    # Use MNE's built-in sample dataset
    sample_data_folder = mne.datasets.sample.data_path()
    raw_fname = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    
    # Load data, pick EEG and EOG channels, filter for clarity
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.pick_types(meg=False, eeg=True, eog=True)
    raw.filter(1, 40, verbose=False) # Standard bandpass
    return raw
 
# Load data state
if 'raw' not in st.session_state:
    with st.spinner('Loading EEG Sample Data...'):
        st.session_state.raw = load_data()
        
        # Detect blinks automatically using EOG to create "Ground Truth"
        eog_events = mne.preprocessing.find_eog_events(st.session_state.raw, verbose=False)
        st.session_state.blink_events = eog_events
 
# --- 3. Game State Management ---
if 'current_start_time' not in st.session_state:
    st.session_state.current_start_time = 10.0  # Start at 10 seconds
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'attempts' not in st.session_state:
    st.session_state.attempts = 0
if 'show_answer' not in st.session_state:
    st.session_state.show_answer = False
 
# --- 4. Helper Functions ---
def get_segment(start_time, duration=5.0):
    raw = st.session_state.raw
    sfreq = raw.info['sfreq']
    start_idx = int(start_time * sfreq)
    stop_idx = int((start_time + duration) * sfreq)
    
    # Get EEG data (Frontal channels are best for blinks: Fp1, Fp2 mostly, here we take a subset)
    picks = ['EEG 001', 'EEG 002', 'EEG 003'] # Indices for typical frontal/temporal
    data, times = raw[picks, start_idx:stop_idx]
    
    # Get EOG data for Ground Truth
    eog_data, _ = raw['EOG 061', start_idx:stop_idx]
    
    return times, data, eog_data, picks
 
def check_for_blink(start_time, duration=5.0):
    # Check if any EOG event falls within the current window
    events = st.session_state.blink_events
    # Events are in samples, convert to time
    event_times = events[:, 0] / st.session_state.raw.info['sfreq']
    
    has_blink = np.any((event_times >= start_time) & (event_times <= start_time + duration))
    return has_blink
 
def next_segment():
    # Jump to a random time or sequential
    duration = 5.0
    max_time = st.session_state.raw.times[-1] - duration
    st.session_state.current_start_time = np.random.uniform(0, max_time)
    st.session_state.show_answer = False
 
# --- 5. Main UI Layout ---
col1, col2 = st.columns([3, 1])
 
with col1:
    st.subheader("Raw EEG Trace")
    
    duration = 3.0 # Show 3 seconds windows
    times, eeg_data, eog_data, ch_names = get_segment(st.session_state.current_start_time, duration)
    
    # Plotting using Matplotlib
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    
    # Plot EEG
    for i in range(len(ch_names)):
        # Offset channels for visibility
        offset = i * 0.0002
        ax1.plot(times, eeg_data[i].T + offset, label=ch_names[i])
    
    ax1.set_title("EEG Channels (Frontal)")
    ax1.set_yticks([]) # Hide y-ticks as amplitudes vary
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Plot Answer (EOG) only if revealed
    if st.session_state.show_answer:
        ax2.plot(times, eog_data[0].T, color='orange', label='EOG (Eye Tracker)')
        
        # Highlight actual blink events
        blink_in_window = check_for_blink(st.session_state.current_start_time, duration)
        if blink_in_window:
            ax2.text(times[len(times)//2], np.mean(eog_data), "BLINK DETECTED",
                     color='red', fontsize=12, ha='center', fontweight='bold')
    else:
        ax2.text(times[len(times)//2], 0, "Hidden (Ground Truth)", ha='center', color='gray')
        
    ax2.set_title("EOG Channel (Ground Truth)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True)
    
    st.pyplot(fig)
 
with col2:
    st.markdown("### Diagnosis")
    st.write(f"Current Window: **{st.session_state.current_start_time:.2f}s**")
    
    # User Controls
    with st.form("diagnosis_form"):
        user_choice = st.radio("Is there a blink in this segment?", ("No Blink", "Yes, there is a Blink"))
        submit_btn = st.form_submit_button("Submit Diagnosis")
    
    if submit_btn and not st.session_state.show_answer:
        st.session_state.attempts += 1
        actual_blink = check_for_blink(st.session_state.current_start_time, duration)
        
        user_said_blink = (user_choice == "Yes, there is a Blink")
        
        if user_said_blink == actual_blink:
            st.success("Correct! 🎉")
            st.session_state.score += 1
        else:
            st.error("Incorrect. ❌")
            
        st.session_state.show_answer = True
        st.rerun()
 
    if st.session_state.show_answer:
        if st.button("Next Segment ➡️"):
            next_segment()
            st.rerun()
 
    st.markdown("---")
    st.metric("Score", f"{st.session_state.score} / {st.session_state.attempts}")
 
# --- 6. Educational Sidebar ---
st.sidebar.title("Training Guide")
st.sidebar.info("""
**How to spot a blink:**
1. **Shape:** Look for a 'U' or 'V' shape (dip or spike depending on polarity).
2. **Location:** Frontal channels (Fp1, Fp2) show it strongest.
3. **Propagation:** It fades as you move to the back of the head.
""")