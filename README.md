# Bank Term Deposit Subscription Prediction
## Banka Vadeli Mevduat Katılım Tahmini

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

---

### 🇬🇧 English Description

#### Project Overview
This project presents a comprehensive data analytics and machine learning approach to predict customer subscription to bank term deposits. Using the **Bank Marketing dataset** from the UCI Machine Learning Repository, we compare multiple classification methods and preprocessing strategies to identify the most effective model for telemarketing campaign optimization.

#### Key Features & Analysis
- **Comparative Modeling:** Decision Tree, Linear Regression (Linear Probability Model), and Random Forest.
- **Preprocessing Strategies:** Comparison between **Imputation** (Mode-based) and **Treating "Unknown" as a distinct category**.
- **In-depth Feature Analysis:** 
    - Demographic factors (Age, Job, Marital Status, Education).
    - Campaign characteristics (Call duration, contact frequency, timing).
    - Macroeconomic indicators (Euribor 3-month rate, Consumer confidence, etc.).
- **Strategic Insights:** Identifies "Golden Segments" such as students and retirees who show significantly higher conversion rates.

#### Results
- **Random Forest** achieved the highest accuracy of **90.99%**.
- **Linear Regression** surprisingly outperformed Decision Tree with **90.15%** accuracy.
- **Call Duration** was identified as the strongest predictor of success.
- Optimized strategies suggest a potential **32.7% increase** in subscription volume.

---

### 🇹🇷 Türkçe Açıklama

#### Proje Hakkında
Bu proje, banka müşterilerinin vadeli mevduat hesaplarına katılım sağlayıp sağlamayacağını tahmin etmek için kapsamlı bir veri analitiği ve makine öğrenmesi yaklaşımı sunmaktadır. UCI Makine Öğrenmesi Deposu'ndan alınan **Banka Pazarlama veri seti** kullanılarak, tele-pazarlama kampanyalarını optimize etmek amacıyla farklı sınıflandırma yöntemleri ve veri ön işleme stratejileri karşılaştırılmıştır.

#### Öne Çıkan Özellikler ve Analizler
- **Karşılaştırmalı Modelleme:** Karar Ağacı (Decision Tree), Doğrusal Regresyon (Lineer Olasılık Modeli) ve Random Forest.
- **Ön İşleme Stratejileri:** **Eksik Veri Atama (Imputation)** ile **"Bilinmiyor" (Unknown) değerini ayrı bir kategori olarak değerlendirme** arasındaki farkların analizi.
- **Derinlemesine Öznitelik Analizi:**
    - Demografik faktörler (Yaş, Meslek, Medeni Durum, Eğitim).
    - Kampanya özellikleri (Arama süresi, iletişim sıklığı, zamanlama).
    - Makroekonomik göstergeler (Euribor 3 aylık faiz oranı, Tüketici güven endeksi vb.).
- **Stratejik Öngörüler:** Öğrenciler ve emekliler gibi dönüşüm oranları anlamlı derecede yüksek olan "Altın Segmentler" belirlenmiştir.

#### Sonuçlar
- **Random Forest** %90.99 ile en yüksek doğruluk oranına ulaşmıştır.
- **Doğrusal Regresyon** şaşırtıcı bir şekilde Karar Ağacı'nı geride bırakarak %90.15 doğruluk sağlamıştır.
- **Arama Süresi (Duration)**, başarının en güçlü göstergesi olarak belirlenmiştir.
- Optimize edilen stratejiler, katılım hacminde **%32.7'lik bir artış** potansiyeline işaret etmektedir.

---

### 🚀 Getting Started / Başlangıç

#### 🛠 Installation / Kurulum
1. Clone the repository / Depoyu klonlayın:
   ```bash
   git clone https://github.com/iamsahinemir/Data-Analytics.git
   ```
2. Install required libraries / Gerekli kütüphaneleri kurun:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

#### 📁 Project Structure / Proje Yapısı
The main project files are located in the `403/` directory:
- `project.ipynb`: Main analysis and model implementations.
- `cross_analysis.py` & `feature_analysis.py`: Modular scripts for data exploration.
- `IEEE_Conference_Paper.md`: Detailed documentation and results formatted as a scientific paper.
- `dataset/`: Contains the Bank Marketing CSV files.

---

### 📊 Visualizations / Görselleştirmeler
The project includes extensive visualizations for:
- Age vs. Subscription rates
- Job and Education impact
- Monthly performance analysis
- Economic indicator correlations

---

### 📜 License / Lisans
This project is open-source and available under the MIT License.

---
*Created by [Emir Sahin](https://github.com/iamsahinemir)*
