# VyomNetra: AI-Enabled Exoplanet Detection Pipeline 🔭

**An end-to-end, automated astronomical pipeline for isolating and characterizing exoplanets from noisy TESS satellite light curves.**

### 🌐 Deployed Application: [Launch VyomNetra Dashboard](https://vyomnetra.streamlit.app)
[![Live Application](https://img.shields.io/badge/Live_Dashboard-Streamlit-FF4B4B?style=for-for-the-badge&logo=streamlit)](https://vyomnetra.streamlit.app)

---

## 📖 Project Overview
**VyomNetra** ("Eye in the Cosmos") is an enterprise-grade scientific software pipeline designed to discover exoplanets by analyzing raw space telemetry from the TESS satellite. 

When a planet orbits a distant star, it occasionally crosses (transits) in front of it, causing a microscopic drop in the star's brightness. Finding these drops is a "needle-in-a-haystack" problem because raw space data is heavily distorted by stellar flares, telescope jitter, and thermal noise. VyomNetra autonomously downloads this raw data, strips away the noise without erasing the physical signals, and utilizes a dual-path mathematical architecture to confirm the presence of an exoplanet, ultimately extracting its precise physical dimensions.

---

## 🚀 Key Technical Innovations (USPs)
While most open-source solutions rely on a single-pass basic algorithm, VyomNetra is engineered to professional TESS/SPOC validation standards. 

1. **Adaptive, Outlier-Aware Detrending:** Utilizes a two-pass windowed median filter combined with Savitzky-Golay smoothing to eradicate instrumental systematics without clipping the genuine U-shaped transit dips.
2. **Dual-Path Search Engine:** Runs a standard Box Least Squares (BLS) periodogram in parallel with a specialized **Mono-Transit Anomaly Scanner**, ensuring the pipeline captures single-event, long-period exoplanets that periodic-only searches miss due to limited observation windows.
3. **Automated Physics-Grounded Vetting:** Replaces manual human inspection with a rigorous false-positive vetting matrix evaluating odd/even transit depth consistency, secondary eclipses, and transit shape/morphology.
4. **Dynamic Astrophysical Extraction:** Autonomously queries the MAST/TIC database to scale signals into precise physical metrics (Planet Radius, Semi-Major Axis, Equilibrium Temperature) based on the Seager & Mallén-Ornelas (2003) framework.

---

## ⚙️ Pipeline Architecture
VyomNetra operates on a strictly decoupled Model-View-Controller (MVC) style architecture, separating the Streamlit UI layer from the core algorithmic computation stack.

1. **Data Ingestion:** Secure fetching of raw SAP Flux (`sap_flux`) telemetry or custom user data via the MAST archive.
2. **Systematics Mitigation:** High-pass filtering, windowed smoothing, and sigma-clipping arrays.
3. **Signal Isolation:** Dual independent paths optimizing both Kovács et al. (2002) Box Least Squares matrix operations and automated single-event anomaly scans.
4. **Phase Folding:** Temporal stacking of the light curve over the discovered orbital period to maximize the Signal-to-Noise Ratio (SNR).
5. **Parameter Extraction:** Deriving physical planet properties and calculating SNR-propagated uncertainties.
6. **Vetting & Output:** Generating an AI-confidence/vetting score, plotting multi-panel diagnostic figures, and exporting structured CSV feature vectors.

---

## 📂 Project Structure

- `core_engine.py` — The heavy-lifting algorithmic backend containing data acquisition, detrending, transit search, parameter extraction, vetting, plotting, and insight generation. Called strictly through public functions.
- `app.py` — The user-facing Streamlit dashboard containing UI configuration, rendering layouts, and reactive elements with zero core algorithmic logic.
- `requirements.txt` — Tracked open-source Python dependencies.

---

## 🛠️ Technology Stack
Implemented in pure Python, VyomNetra utilizes a robust scientific computing stack with zero proprietary dependencies:
* **UI & Deployment:** Streamlit
* **Core Computing:** NumPy, Pandas, SciPy (Signal processing)
* **Astrophysics Math:** Astropy (BLS periodograms, physical constants)
* **Space Data Acquisition:** Lightkurve, Astroquery (MAST/TIC catalog routing)
* **Diagnostics:** Matplotlib (High-resolution phase-folded visualizations)

---

## 🏃 Running the Application Locally

Install dependencies and run the dashboard server with the following:

```bash
pip install -r requirements.txt
streamlit run app.py
