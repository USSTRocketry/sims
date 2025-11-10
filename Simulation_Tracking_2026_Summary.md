# 🚀 Simulation Tracking — LC2026 Competition Rocket

This document summarizes simulation progress for the **USST LC2026 competition rocket**, including both **OpenRocket** and **RasAero II** test cases.  
It serves as a changelog and performance comparison log to track geometry, mass distribution, and stability evolution throughout design iterations.

---

## 🧩 OpenRocket Simulations

| **Simulation Name** | **Description** | **Original File** | **Created By** | **Date Created** | **Checked By** | **Date Checked** | **Min Stability (%)** | **Max Stability (%)** | **Velocity Off Rail (m/s)** | **Max Velocity (m/s)** | **Apogee (ft)** | **Mass (kg)** | **Comments** |
|----------------------|----------------|------------------|----------------|------------------|----------------|------------------|--------------------------|--------------------------|------------------------------|------------------------|----------------|----------------|---------------|
| `2026Rocket_v1.0.ork` | First simulation with new length, updated component locations, and revised mass. Fins reused from previous “Up” configuration. | — | Quinn Lawson | 2025-10-13 | Nishok Deenadayalan | 2025-10-15 | - | - | 38 | 403 | 13 000 | 23.44 | — |
| `2026Rocket_v1.1.ork` | Swept Delta fin configuration | `2026Rocket_v1.0` | Quinn Lawson | 2025-11-08 | - | - | 16.5 | 19.5 | 38 | 404 | 13293 | 23.4 | — |
| `2026Rocket_v1.2.ork` | Larger Trapezoidal fin configuration | `2026Rocket_v1.0` | Quinn Lawson | 2025-11-09 | - | - | 16.4 | 18.8 | 38 | 400 | 12493 | 23.4 | — |
| `2026Rocket_v2.0.ork` | Derived from v1.0, but **without the boat-tail**. | `2026Rocket_v1.0` | Nishok Deenadayalan | 2025-10-15 | — | — | - | - | 37.1 | 406 | 12 750 | 23.13 | Over-stable |

### **Summary of OpenRocket Progress**
- v1.0 introduced revised airframe length and mass distribution; stability remained within design limits.  
- v2.0 removed the boat-tail, increasing static margin (over-stability observed).  
- Slight decrease in velocity off the rail (−0.9 m/s) and minor apogee change (~−250 ft).  
- Overall performance stable across revisions.

---

## 🧮 RasAero II Simulations

| **Simulation Name** | **Description** | **Original File** | **Created By** | **Date Created** | **Checked By** | **Max Stability (Cal)** | **Velocity Off Rail (m/s)** | **Max Velocity (m/s)** | **Apogee (ft)** | **Mass (lb)** | **Comments** |
|----------------------|----------------|------------------|----------------|------------------|----------------|--------------------------|------------------------------|------------------------|----------------|----------------|---------------|
| `2026Rocket_v2.0.CDX1` | First RasAero simulation using geometry from `2026Rocket_v2.0.ork`. Initial baseline run. | — | Nishok Deenadayalan | 2025-10-15 | — | 50.99 | — | — | — | — | Requires refinement; work in progress. |

### **Summary of RasAero Progress**
- First RasAero validation run using identical geometry as OpenRocket v2.0.  
- Stability index extremely high (50.99 cal), indicating likely configuration error or placeholder coefficient scaling.  
- Model requires further tuning and drag coefficient verification.

---

## 📘 Overall Change Log Summary

| **Version** | **Major Change** | **Impact** |
|--------------|------------------|-------------|
| v1.0 → v2.0 | Boat-tail removed | Increased static margin → over-stable; slight loss in rail velocity and apogee. |
| v2.0 (OpenRocket) → v2.0.CDX1 (RasAero) | Switched simulation software | Geometry consistent, stability scaling under verification. |

---

## 🧠 Notes & Next Steps
- Review **RasAero setup** (fin dimensions, nosecone coefficient tables, and launch rail conditions).  
- Validate **mass consistency** between OR and RasAero models.  
- Adjust **boat-tail geometry** in future versions to find stability trade-off between 2.0–2.5 Cal.  
- Continue tracking with consistent naming convention: `2026Rocket_vX.Y`.
