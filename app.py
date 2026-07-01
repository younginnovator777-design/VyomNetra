"""
================================================================================
 VyomNetra  ::  AI-Assisted Exoplanet Transit Detection Dashboard
================================================================================
Streamlit UI layer. 
"""

import io

import pandas as pd
import streamlit as st

import core_engine as engine

st.set_page_config(
    page_title="VyomNetra | Exoplanet Detection",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "tic_label" not in st.session_state:
    st.session_state.tic_label = None
if "show_about" not in st.session_state:
    st.session_state.show_about = False
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# ------------------------------------------------------------------------
# Theme CSS
# ------------------------------------------------------------------------
DARK_CSS = """
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .stApp { background-color: #0D1117; }
    section[data-testid="stSidebar"] { background-color: #0B0E14; border-right: 1px solid #21262D; }
    h1,h2,h3,h4 { color: #E6EDF3 !important; }
    p,li,span,label,.stMarkdown { color: #C9D1D9; }
    .card { background:#161B22; border:1px solid #30363D; border-radius:10px; padding:16px 18px; margin-bottom:10px; }
    .insight-positive { border-left:3px solid #10B981; background:#0F2119; padding:10px 14px; border-radius:6px; margin-bottom:8px; color:#D9F5E6; }
    .insight-warning  { border-left:3px solid #F59E0B; background:#241B08; padding:10px 14px; border-radius:6px; margin-bottom:8px; color:#FBE8C6; }
    .insight-neutral  { border-left:3px solid #6366F1; background:#161A2E; padding:10px 14px; border-radius:6px; margin-bottom:8px; color:#DCE0FA; }
    .badge-pass { color:#10B981; font-weight:700; } .badge-fail { color:#EF4444; font-weight:700; } .badge-neutral { color:#F59E0B; font-weight:700; }
    .asv-title { font-size:2.1rem; font-weight:800; color:#E6EDF3; letter-spacing:-0.5px; }
    .asv-sub { color:#8B949E; font-size:0.95rem; margin-top:-8px; }
    div.stButton > button { background-color:#1F6FEB; color:white; border:none; border-radius:6px; font-weight:600; }
    div.stButton > button:hover { background-color:#388BFD; }
    .stTabs [data-baseweb="tab"] { color:#8B949E; } .stTabs [aria-selected="true"] { color:#E6EDF3 !important; }
    [data-testid="stMetricValue"] { color: #E6EDF3; }
</style>
"""

LIGHT_CSS = """
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .stApp { background-color: #F6F8FA; }
    section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #D0D7DE; }
    h1,h2,h3,h4 { color: #1F2328 !important; }
    p,li,span,label,.stMarkdown { color: #24292F; }
    .card { background:#FFFFFF; border:1px solid #D0D7DE; border-radius:10px; padding:16px 18px; margin-bottom:10px; }
    .insight-positive { border-left:3px solid #1A7F37; background:#DAFBE1; padding:10px 14px; border-radius:6px; margin-bottom:8px; color:#0F5323; }
    .insight-warning  { border-left:3px solid #9A6700; background:#FFF8C5; padding:10px 14px; border-radius:6px; margin-bottom:8px; color:#5A4400; }
    .insight-neutral  { border-left:3px solid #4C51E6; background:#EEF0FE; padding:10px 14px; border-radius:6px; margin-bottom:8px; color:#2A2E80; }
    .badge-pass { color:#1A7F37; font-weight:700; } .badge-fail { color:#CF222E; font-weight:700; } .badge-neutral { color:#9A6700; font-weight:700; }
    .asv-title { font-size:2.1rem; font-weight:800; color:#1F2328; letter-spacing:-0.5px; }
    .asv-sub { color:#57606A; font-size:0.95rem; margin-top:-8px; }
    div.stButton > button { background-color:#0969DA; color:white; border:none; border-radius:6px; font-weight:600; }
    div.stButton > button:hover { background-color:#218BFF; }
    .stTabs [data-baseweb="tab"] { color:#57606A; } .stTabs [aria-selected="true"] { color:#1F2328 !important; }
</style>
"""

st.markdown(DARK_CSS if st.session_state.theme == "Dark" else LIGHT_CSS, unsafe_allow_html=True)


def insight_block(kind, text):
    cls = {"positive": "insight-positive", "warning": "insight-warning", "neutral": "insight-neutral"}[kind]
    icon = {"positive": "✅", "warning": "⚠️", "neutral": "ℹ️"}[kind]
    st.markdown(f"<div class='{cls}'>{icon} {text}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='asv-title'>🪐 VyomNetra</div>", unsafe_allow_html=True)
    st.markdown("<div class='asv-sub'>Dual-path exoplanet transit detection</div>", unsafe_allow_html=True)

    top_c1, top_c2 = st.columns([1.3, 1])
    with top_c1:
        if st.button("ℹ️ About This Project", use_container_width=True):
            st.session_state.show_about = not st.session_state.show_about
    with top_c2:
        st.session_state.theme = st.selectbox("Theme", ["Dark", "Light"],
                                                index=0 if st.session_state.theme == "Dark" else 1,
                                                label_visibility="collapsed")

    st.markdown("---")

    mode = st.radio(
        "Data mode", ["Demo Run (WASP-19 b)", "Custom Target"],
        help="Demo Run replays the fixed WASP-19 b (TIC 35516889) reference case. "
             "Custom Target searches a real TESS TIC ID or your own uploaded light curve."
    )
    st.markdown("---")

    if mode.startswith("Demo"):
        st.caption(
            "**WASP-19 b** — ultra-short-period hot Jupiter (P ≈ 0.79 d), "
            "used as the fixed validation benchmark for this pipeline."
        )
        run_clicked = st.button("▶ Run Demo Pipeline", use_container_width=True)
        source_mode, tic_input, data_file = "demo", "WASP-19 b (TIC 35516889)", None

    else:
        data_source = st.radio("Data source", ["Fetch from TESS / MAST", "Upload light curve CSV"])

        if data_source == "Fetch from TESS / MAST":
            tic_input = st.text_input("TIC ID", value="TIC 35516889",
                                        help="Must be resolvable on the MAST archive, e.g. 'TIC 35516889'")
            source_mode, data_file = "mast", None
        else:
            tic_input = st.text_input("Target label (for display)", value="CUSTOM-UPLOAD")
            data_file = st.file_uploader("CSV with columns: time, flux[, flux_err]", type=["csv"])
            source_mode = "csv"

        with st.expander("Search parameters", expanded=True):
            period_min = st.number_input("Min period (days)", 0.1, 50.0, 0.5, 0.1)
            period_max = st.number_input("Max period (days)", 0.5, 100.0, 13.0, 0.5)
            n_periods = st.slider("Period grid resolution", 1000, 10000, 5000, 500)
            snr_threshold = st.number_input("SNR detection threshold", 1.0, 20.0, 7.0, 0.5)

        with st.expander("Host star parameters (optional)"):
            st.caption("Defaults auto-query the TESS Input Catalog. Override manually if you know them.")
            use_manual_star = st.checkbox("Override with manual stellar parameters")
            r_star = st.number_input("R_star (solar radii)", 0.1, 20.0, 1.0, 0.1) if use_manual_star else None
            m_star = st.number_input("M_star (solar masses)", 0.1, 20.0, 1.0, 0.1) if use_manual_star else None
            t_star = st.number_input("T_eff (K)", 2000, 50000, 5778, 50) if use_manual_star else None

        run_clicked = st.button("▶ Run Detection Pipeline", use_container_width=True)


# ------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------
st.markdown("<div class='asv-title'>VyomNetra</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='asv-sub'>Dual-path pipeline for periodic and single-event transit detection in TESS photometry</div>",
    unsafe_allow_html=True,
)
st.write("")

# ------------------------------------------------------------------------
# About panel
# ------------------------------------------------------------------------
if st.session_state.show_about:
    st.markdown("## About VyomNetra")
    st.markdown(
        "VyomNetra searches TESS photometry for the periodic brightness dips caused by "
        "transiting exoplanets, extracts physical planet parameters from the transit geometry, "
        "and automatically screens out common false positives — without manual light-curve inspection."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### What makes it different (USP)")
        st.markdown(
            """
- **Dual-path detection engine** — most pipelines run only a periodic
  search, which misses planets that transited just once in the observing
  window (long-period or eccentric orbits). VyomNetra runs a **second,
  independent anomaly scan** specifically to catch these single
  "mono-transit" events, so nothing is missed just because it didn't repeat.
- **Automated false-positive vetting** — every candidate is screened
  through three tests (odd/even depth, secondary eclipse, transit shape)
  before being labelled, catching eclipsing binaries a depth-only
  threshold would wrongly report as planets.
- **Findings-first reporting** — the dashboard surfaces what the model
  concluded and why, instead of dumping raw arrays the user has to
  interpret themselves.
- **Works online or offline** — pulls live TESS data from MAST, or runs
  on a fixed, fully reproducible benchmark (WASP-19 b) for demonstration.
- **Physically grounded** — planet radius, orbital distance, and
  equilibrium temperature are derived from named, published relations,
  not a fitted regression.
            """
        )
    with c2:
        st.markdown("#### Pipeline stages")
        st.markdown(
            """
1. **Data acquisition** — TESS 2-min cadence SAP flux from MAST, or a
   user-supplied light curve.
2. **Detrending** — two-pass filter (coarse median filter, then
   sigma-clipped Savitzky-Golay smoothing) removes systematics while
   preserving transit shapes.
3. **Periodic transit search** — Box Least Squares (Kovács et al. 2002),
   the same core algorithm used in the official TESS/SPOC pipeline.
4. **Mono-transit anomaly scan** — an independent peak-detection pass on
   inverted, detrended flux flags isolated dips that never repeat.
5. **Phase folding** — light curve folded on the best period and binned.
6. **Parameter extraction** — planet radius, semi-major axis, and
   equilibrium temperature via Seager & Mallén-Ornelas (2003) and
   Kepler's third law.
7. **False-positive vetting** — odd/even, secondary eclipse, and
   transit-shape tests separate planets from binaries.
8. **Insight generation** — translates the raw numbers into plain-language
   findings, highlighting where the dual-path design added value.
            """
        )
    st.info(
        "Detection significance is reported as **SNR** (depth vs. photometric scatter) "
        "and **SDE** (how far the best periodogram peak stands above the noise floor), "
        "combined with the vetting outcome into one confidence percentage."
    )
    st.markdown("---")


# ------------------------------------------------------------------------
# Run pipeline
# ------------------------------------------------------------------------
if run_clicked:
    try:
        with st.spinner("Running VyomNetra pipeline..."):
            r_star = m_star = t_star = None

            if mode.startswith("Demo"):
                t, f, fe, meta = engine.generate_demo_lightcurve()
                tic_label = meta["target_name"]
                config = {"period_min": 0.4, "period_max": 3.0, "n_periods": 7000, "snr_threshold": 7.0}

            else:
                config = {"period_min": period_min, "period_max": period_max,
                          "n_periods": int(n_periods), "snr_threshold": snr_threshold}
                if use_manual_star:
                    r_star, m_star, t_star = r_star, m_star, t_star

                if source_mode == "mast":
                    t, f, fe = engine.download_tess_lc(tic_input)
                    tic_label = tic_input
                else:
                    if data_file is None:
                        st.error("Please upload a CSV file with 'time' and 'flux' columns.")
                        st.stop()
                    df_in = pd.read_csv(data_file)
                    t, f, fe = engine.load_custom_csv(df_in)
                    tic_label = tic_input

            result = engine.run_pipeline(t, f, fe, tic_label, config=config,
                                          r_star=r_star, m_star=m_star, t_star=t_star)
            st.session_state.result = result
            st.session_state.tic_label = tic_label

    except Exception as e:
        st.error(f"Pipeline run failed: {e}")
        if mode == "Custom Target" and source_mode == "mast":
            st.warning(
                "Live MAST download failed — the target may be unresolvable, or the network "
                "is unreachable from this environment. Try Demo Run, or upload your own CSV instead."
            )
        st.session_state.result = None


# ------------------------------------------------------------------------
# Results
# ------------------------------------------------------------------------
result = st.session_state.result

if result is None:
    st.markdown(
        "Run the **Demo (WASP-19 b)** case for an instant validated example, "
        "or configure a **Custom Target** in the sidebar — then press run."
    )
else:
    params, vetting, bls = result["params"], result["vetting"], result["bls_result"]
    label = vetting["label"]
    badge_class = "badge-pass" if label == "PLANET CANDIDATE" else (
        "badge-neutral" if ("UNCERTAIN" in label or "SUB-THRESHOLD" in label) else "badge-fail")

    st.markdown(f"### Result for `{st.session_state.tic_label}`")
    st.markdown(f"<span class='{badge_class}' style='font-size:1.3rem;'>{label}</span>",
                unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Period (days)", f"{params['period_days']:.4f}")
    m2.metric("Depth (ppm)", f"{params['depth_ppm']:.0f}")
    m3.metric("Duration (hr)", f"{params['duration_hr']:.2f}")
    m4.metric("Radius (R⊕)", f"{params['Rp_Rearth']:.2f}")
    m5.metric("SNR", f"{params['snr']:.1f}")
    m6.metric("Confidence", f"{vetting['confidence_pct']:.1f}%")

    st.write("")

    # ---- Diagnostic figure: the primary model output, shown up front ----
    st.markdown("#### 📈 Diagnostic Figure")
    st.caption("Generated directly by the model for this run — raw & detrended flux, "
               "BLS periodogram, phase-folded transit, and parameter summary.")
    st.pyplot(result["figure"], use_container_width=True)

    st.write("")
    st.markdown("#### 🧠 What the model found")
    st.caption("Findings, not raw arrays — this is the model's interpretation of the run, "
               "including exactly where the dual-path design added value over a standard periodic-only search.")
    for item in result["insights"]:
        insight_block(item["kind"], item["text"])

    st.write("")
    tab_findings, tab_vetting, tab_mono, tab_export = st.tabs(
        ["📊 Key Results Table", "🔍 FP Vetting", "🌗 Mono-Transit Scan", "⬇️ Export"]
    )

    with tab_findings:
        st.markdown("##### Summary parameters")
        st.dataframe(result["summary_table"], use_container_width=True, hide_index=True)
        st.markdown("##### Phase-folded / binned transit profile")
        st.caption("Compact, binned representation of the transit — not the raw cadence-by-cadence flux.")
        st.dataframe(result["phase_table"], use_container_width=True, height=220)

    with tab_vetting:
        st.markdown("##### False-positive vetting results")
        v1, v2, v3 = st.columns(3)
        for col, name, passed, detail in [
            (v1, "Odd/Even Depth", vetting["test1_pass"], f"σ = {vetting['odd_even_sig']:.2f}"),
            (v2, "Secondary Eclipse", vetting["test2_pass"], f"{vetting['sec_depth_ppm']:.0f} ppm"),
            (v3, "Transit Shape (U vs V)", vetting["test3_pass"], f"flat frac = {vetting['flat_frac']:.2f}"),
        ]:
            with col:
                st.markdown(f"**{name}**")
                cls = "badge-pass" if passed else "badge-fail"
                st.markdown(f"<span class='{cls}'>{'PASS' if passed else 'FAIL'}</span> — {detail}",
                            unsafe_allow_html=True)
        st.markdown("##### ML-ready feature vector")
        st.json(vetting["ml_features"])

    with tab_mono:
        st.markdown("##### Mono-transit anomaly scan (dual-path detection)")
        st.caption(
            "Runs independently of the periodic BLS search — flags isolated dips that never "
            "repeat within the observing baseline, so single-transit, long-period candidates aren't missed."
        )
        if len(result["mono_table"]) == 0:
            st.info("No significant isolated anomalies found above the detection threshold.")
        else:
            st.dataframe(result["mono_table"], use_container_width=True, hide_index=True)

    with tab_export:
        st.caption("Full-resolution light curve data is generated on demand here — kept out of "
                   "the on-screen tables above to keep the dashboard fast and focused on findings.")
        csv_buf = io.StringIO()
        result["summary_table"].to_csv(csv_buf, index=False)
        st.download_button("Download summary (CSV)", csv_buf.getvalue(),
                            file_name=f"astro_sieve_summary_{st.session_state.tic_label.replace(' ', '_')}.csv",
                            mime="text/csv")

        lc_buf = io.StringIO()
        result["lc_table"].to_csv(lc_buf, index=False)
        st.download_button("Download full light curve data (CSV)", lc_buf.getvalue(),
                            file_name=f"astro_sieve_lightcurve_{st.session_state.tic_label.replace(' ', '_')}.csv",
                            mime="text/csv")

        img_buf = io.BytesIO()
        result["figure"].savefig(img_buf, format="png", dpi=150, bbox_inches="tight",
                                  facecolor=result["figure"].get_facecolor())
        st.download_button("Download diagnostic figure (PNG)", img_buf.getvalue(),
                            file_name=f"astro_sieve_diagnostic_{st.session_state.tic_label.replace(' ', '_')}.png",
                            mime="image/png")

st.markdown("---")
st.caption("VyomNetra — AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves")
