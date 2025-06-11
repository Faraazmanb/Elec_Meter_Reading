# 🔬 Automated Electricity Meter Reading System  
**Research Project by Mohamed Faraazman Bin Farooq S**  
*Final Year B.Tech in AI & Data Science, BSA Crescent University*

---

## 📍 Overview

Manual electricity meter reading is still prevalent in many parts of the world, including Tamil Nadu, leading to challenges such as:

- ❌ Human errors in data entry  
- 🚪 Missed readings when residents aren't home  
- 🕒 Inefficient and time-consuming processes  
- 🧾 Delays in billing and frequent disputes

This research addresses these issues by proposing a lightweight computer vision and OCR-based pipeline to **automate electricity meter reading** using image-based input.

---

## 🎯 Objectives

- Automatically extract readings from digital electricity meter images  
- Enhance OCR accuracy through preprocessing  
- Build a deployable prototype using open-source tools  
- Evaluate robustness under real-world conditions

---

## 🛠️ Technologies Used

- **Python 3.9+**  
- **OpenCV** – Image preprocessing & contour detection  
- **Tesseract OCR** – Optical character recognition  
- **NumPy / Pandas** – Data manipulation  
- **Streamlit (optional)** – UI for prototype demo  

---

## 🧪 Methodology

1. **Data Collection**
   - Curated 100+ electricity meter images from real-world scenarios
   - Labeled true readings for validation

2. **Image Preprocessing**
   - Grayscale conversion  
   - Adaptive thresholding  
   - Morphological operations for noise removal

3. **Region of Interest (ROI) Detection**
   - Contour detection to isolate digital display  
   - Cropping and alignment for OCR optimization

4. **OCR Processing**
   - Applied Tesseract OCR to extract digits  
   - Post-processing cleanup to improve reliability

5. **Results Logging**
   - Exported readings with timestamp and image ID to CSV

---

## 📊 Experimental Results

| Metric                    | Value     |
|--------------------------|-----------|
| Accuracy (clear images)  | ~92%      |
| Accuracy (tilted/shadow) | ~81%      |
| Avg. processing time     | < 1.5 sec |
| Dataset size             | 100+ images |

---

## 🌐 Contributions

- ✅ Developed a complete CV + OCR pipeline for meter reading  
- ✅ Created dataset and labeling for reproducible experiments  
- ✅ Demonstrated high accuracy with minimal resources  
- ✅ Prepared baseline for future deep learning enhancements

---

## 🔮 Future Enhancements

- Replace OCR with CNN-based digit detection  
- Build Android/iOS capture app with real-time reading  
- Extend support to analog meters  
- Integrate with cloud databases and utility dashboards

---

## 🧠 Research Impact

This project showcases how low-cost AI solutions can enable **smart utility automation** without requiring infrastructure overhaul. The methodology bridges the gap between traditional systems and smart grids, especially in semi-urban and rural areas.

---

## 📁 Project Structure

