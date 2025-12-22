import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from scipy.stats import chi2_contingency

rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Veri setini yükle
print("="*80)
print("ÇAPRAZ ANALİZ VE DERİN İÇGÖRÜLER")
print("="*80 + "\n")

data = pd.read_csv('dataset/bank-additional-full.csv', sep=';')
data = data.drop_duplicates()

# Yaş grupları oluştur
data['age_group'] = pd.cut(data['age'], 
                            bins=[0, 25, 35, 45, 55, 65, 100],
                            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# ============================================================================
# 1. MEDENİ DURUM x YAŞ ETKİLEŞİMİ
# ============================================================================
print("\n1. MEDENİ DURUM x YAŞ ETKİLEŞİMİ ANALİZİ")
print("-" * 80)

# Medeni durum ve yaş grubu bazında abonelik oranları
marital_age_analysis = pd.crosstab([data['marital'], data['age_group']], 
                                    data['y'], normalize='index') * 100

print("\nMedeni Durum ve Yaş Grubuna Göre Abonelik Oranları (%):")
print(marital_age_analysis['yes'].unstack().round(2))

# En yüksek ve en düşük kombinasyonlar
marital_age_flat = marital_age_analysis['yes'].reset_index()
marital_age_flat.columns = ['marital', 'age_group', 'subscription_rate']
marital_age_flat = marital_age_flat.sort_values('subscription_rate', ascending=False)

print("\n🏆 EN YÜKSEK ABONELİK ORANLARI (Medeni Durum x Yaş):")
print(marital_age_flat.head(10).to_string(index=False))

print("\n❌ EN DÜŞÜK ABONELİK ORANLARI (Medeni Durum x Yaş):")
print(marital_age_flat.tail(10).to_string(index=False))

# İstatistiksel test - Chi-square
contingency_table = pd.crosstab([data['marital'], data['age_group']], data['y'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"\n📊 Chi-Square Test: χ² = {chi2:.2f}, p-value = {p_value:.4f}")
if p_value < 0.05:
    print("✅ Medeni durum ve yaş grubu, abonelik ile İSTATİSTİKSEL OLARAK ANLAMLI ilişkili!")
else:
    print("❌ İstatistiksel olarak anlamlı ilişki bulunamadı.")

# Görselleştirme
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1.1 Heatmap
pivot_marital_age = marital_age_analysis['yes'].unstack()
sns.heatmap(pivot_marital_age, annot=True, fmt='.1f', cmap='YlGnBu', 
            ax=axes[0, 0], cbar_kws={'label': 'Abonelik Orani (%)'})
axes[0, 0].set_title('Medeni Durum x Yas Grubu - Abonelik Orani Heatmap', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Yas Grubu')
axes[0, 0].set_ylabel('Medeni Durum')

# 1.2 Grouped bar chart
pivot_marital_age.T.plot(kind='bar', ax=axes[0, 1], colormap='Set2')
axes[0, 1].set_title('Yas Grubuna Gore Medeni Durum Karsilastirmasi', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Yas Grubu')
axes[0, 1].set_ylabel('Abonelik Orani (%)')
axes[0, 1].legend(title='Medeni Durum')
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

yes_data = data[data['y'] == 'yes']
for marital in ['married', 'single', 'divorced']:
    subset = yes_data[yes_data['marital'] == marital]['age']
    axes[1, 0].hist(subset, alpha=0.5, label=marital, bins=20)
axes[1, 0].set_title('Abonelik Yapanlarda Medeni Duruma Gore Yas Dagilimi', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Yas')
axes[1, 0].set_ylabel('Frekans')
axes[1, 0].legend()

avg_age_marital = data.groupby(['marital', 'y'])['age'].mean().unstack()
avg_age_marital.plot(kind='bar', ax=axes[1, 1], color=['red', 'green'])
axes[1, 1].set_title('Medeni Duruma Gore Ortalama Yas (Abonelik Durumuna Gore)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Medeni Durum')
axes[1, 1].set_ylabel('Ortalama Yas')
axes[1, 1].legend(['No', 'Yes'])
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('cross_analysis_marital_age.png', dpi=300, bbox_inches='tight')
print("\n✓ Grafik kaydedildi: cross_analysis_marital_age.png")
plt.close()

print("\n" + "="*80 + "\n")

print("\n2. MESLEK x EĞİTİM x YAŞ KOMBİNASYONU ANALİZİ")
print("-" * 80)

top_jobs = data['job'].value_counts().head(8).index

job_edu_analysis = pd.crosstab([data[data['job'].isin(top_jobs)]['job'], 
                                 data[data['job'].isin(top_jobs)]['education']], 
                                data[data['job'].isin(top_jobs)]['y'], 
                                normalize='index') * 100

print("\nMeslek ve Eğitim Seviyesine Göre Abonelik Oranları (%):")
job_edu_flat = job_edu_analysis['yes'].reset_index()
job_edu_flat.columns = ['job', 'education', 'subscription_rate']
job_edu_flat = job_edu_flat.sort_values('subscription_rate', ascending=False)
print(job_edu_flat.head(15).to_string(index=False))

job_edu_age = pd.crosstab([data[data['job'].isin(top_jobs[:5])]['job'],
                            data[data['job'].isin(top_jobs[:5])]['education'],
                            data[data['job'].isin(top_jobs[:5])]['age_group']], 
                           data[data['job'].isin(top_jobs[:5])]['y'], 
                           normalize='index') * 100

job_edu_age_flat = job_edu_age['yes'].reset_index()
job_edu_age_flat.columns = ['job', 'education', 'age_group', 'subscription_rate']
job_edu_age_flat = job_edu_age_flat.sort_values('subscription_rate', ascending=False)

print("\n🎯 EN YÜKSEK DÖNÜŞÜM KOMBİNASYONLARI (Meslek x Eğitim x Yaş):")
print(job_edu_age_flat.head(20).to_string(index=False))

print("\n📊 Meslek Bazında Ortalama Yaş ve Eğitim Dağılımı:")
for job in top_jobs[:5]:
    job_data = data[data['job'] == job]
    avg_age = job_data['age'].mean()
    edu_mode = job_data['education'].mode()[0]
    subscription_rate = (job_data['y'] == 'yes').mean() * 100
    print(f"{job:20s} | Ort. Yaş: {avg_age:5.1f} | En Yaygın Eğitim: {edu_mode:20s} | Abonelik: {subscription_rate:5.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

job_edu_pivot = job_edu_analysis.loc[top_jobs[:5]]['yes'].unstack()
sns.heatmap(job_edu_pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
            ax=axes[0, 0], cbar_kws={'label': 'Abonelik Orani (%)'})
axes[0, 0].set_title('Meslek x Egitim Seviyesi - Abonelik Orani', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Egitim Seviyesi')
axes[0, 0].set_ylabel('Meslek')

for job in top_jobs[:4]:
    subset = data[data['job'] == job]['age']
    axes[0, 1].hist(subset, alpha=0.4, label=job, bins=25)
axes[0, 1].set_title('Mesleklere Gore Yas Dagilimi', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Yas')
axes[0, 1].set_ylabel('Frekans')
axes[0, 1].legend()

edu_age_analysis = pd.crosstab([data['education'], data['age_group']], 
                                data['y'], normalize='index') * 100
edu_age_pivot = edu_age_analysis['yes'].unstack()
edu_age_pivot_top = edu_age_pivot.loc[['university.degree', 'high.school', 'basic.9y', 'professional.course']]
sns.heatmap(edu_age_pivot_top, annot=True, fmt='.1f', cmap='Blues', 
            ax=axes[1, 0], cbar_kws={'label': 'Abonelik Orani (%)'})
axes[1, 0].set_title('Egitim Seviyesi x Yas Grubu - Abonelik Orani', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Yas Grubu')
axes[1, 0].set_ylabel('Egitim Seviyesi')

top_combos = job_edu_age_flat.head(15).copy()
top_combos['label'] = top_combos['job'] + '\n' + top_combos['education'] + '\n' + top_combos['age_group'].astype(str)
axes[1, 1].barh(range(len(top_combos)), top_combos['subscription_rate'], color='teal')
axes[1, 1].set_yticks(range(len(top_combos)))
axes[1, 1].set_yticklabels(top_combos['label'], fontsize=8)
axes[1, 1].set_xlabel('Abonelik Orani (%)')
axes[1, 1].set_title('En Yuksek Donusum Kombinasyonlari (Top 15)', fontsize=14, fontweight='bold')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('cross_analysis_job_edu_age.png', dpi=300, bbox_inches='tight')
print("\n✓ Grafik kaydedildi: cross_analysis_job_edu_age.png")
plt.close()

print("\n" + "="*80 + "\n")

print("\n3. KAMPANYA ÖZELLİKLERİ x DEMOGRAFİK ÖZELLİKLER")
print("-" * 80)

data['duration_group'] = pd.cut(data['duration'], 
                                bins=[0, 120, 300, 600, 5000],
                                labels=['Kisa (<2dk)', 'Orta (2-5dk)', 'Uzun (5-10dk)', 'Cok Uzun (>10dk)'])

duration_age = pd.crosstab([data['duration_group'], data['age_group']], 
                           data['y'], normalize='index') * 100

print("\nArama Süresi ve Yaş Grubuna Göre Abonelik Oranları (%):")
print(duration_age['yes'].unstack().round(2))

duration_job = pd.crosstab([data[data['job'].isin(top_jobs[:5])]['duration_group'], 
                            data[data['job'].isin(top_jobs[:5])]['job']], 
                           data[data['job'].isin(top_jobs[:5])]['y'], 
                           normalize='index') * 100

print("\nArama Süresi ve Mesleğe Göre Abonelik Oranları (%):")
print(duration_job['yes'].unstack().round(2))

print("\n📊 Önceki Kampanya Sonucu x Demografik Özellikler:")

poutcome_age = pd.crosstab([data['poutcome'], data['age_group']], 
                           data['y'], normalize='index') * 100
print("\nÖnceki Kampanya x Yaş Grubu:")
print(poutcome_age['yes'].unstack().round(2))

poutcome_edu = pd.crosstab([data['poutcome'], data['education']], 
                           data['y'], normalize='index') * 100

poutcome_edu_yes = poutcome_edu['yes'].reset_index()
poutcome_edu_filtered = poutcome_edu_yes[poutcome_edu_yes['education'].isin(['university.degree', 'high.school', 'basic.9y'])]
poutcome_edu_pivot = poutcome_edu_filtered.pivot(index='poutcome', columns='education', values='yes')
print("\nÖnceki Kampanya x Eğitim Seviyesi (Top 3):")
print(poutcome_edu_pivot.round(2))

# Contact type x Demografik
contact_age = pd.crosstab([data['contact'], data['age_group']], 
                          data['y'], normalize='index') * 100
print("\nİletişim Türü x Yaş Grubu:")
print(contact_age['yes'].unstack().round(2))

# Görselleştirme
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 3.1 Duration x Yaş heatmap
duration_age_pivot = duration_age['yes'].unstack()
sns.heatmap(duration_age_pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
            ax=axes[0, 0], cbar_kws={'label': 'Abonelik Orani (%)'})
axes[0, 0].set_title('Arama Suresi x Yas Grubu', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Yas Grubu')
axes[0, 0].set_ylabel('Arama Suresi')

# 3.2 Duration x Meslek
duration_job_pivot = duration_job['yes'].unstack()
duration_job_pivot.plot(kind='bar', ax=axes[0, 1], colormap='tab10')
axes[0, 1].set_title('Arama Suresi x Meslek', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Arama Suresi')
axes[0, 1].set_ylabel('Abonelik Orani (%)')
axes[0, 1].legend(title='Meslek', bbox_to_anchor=(1.05, 1))
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')

# 3.3 Poutcome x Yaş
poutcome_age_pivot = poutcome_age['yes'].unstack()
poutcome_age_pivot.T.plot(kind='bar', ax=axes[0, 2], colormap='Set1')
axes[0, 2].set_title('Onceki Kampanya Sonucu x Yas Grubu', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Yas Grubu')
axes[0, 2].set_ylabel('Abonelik Orani (%)')
axes[0, 2].legend(title='Poutcome')
axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), rotation=45)

# 3.4 Contact x Yaş
contact_age_pivot = contact_age['yes'].unstack()
contact_age_pivot.T.plot(kind='bar', ax=axes[1, 0], color=['steelblue', 'coral'])
axes[1, 0].set_title('Iletisim Turu x Yas Grubu', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Yas Grubu')
axes[1, 0].set_ylabel('Abonelik Orani (%)')
axes[1, 0].legend(title='Contact')
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)

# 3.5 Duration dağılımı (yes vs no)
yes_duration = data[data['y'] == 'yes']['duration']
no_duration = data[data['y'] == 'no']['duration']
axes[1, 1].hist([no_duration, yes_duration], bins=50, label=['No', 'Yes'], 
                alpha=0.7, color=['red', 'green'], range=(0, 1500))
axes[1, 1].set_xlabel('Arama Suresi (saniye)')
axes[1, 1].set_ylabel('Frekans')
axes[1, 1].set_title('Arama Suresi Dagilimi (Abonelik Durumuna Gore)', fontsize=12, fontweight='bold')
axes[1, 1].legend()

# 3.6 Campaign sayısı x Yaş grubu
campaign_age = data.groupby(['age_group', 'y'])['campaign'].mean().unstack()
campaign_age.plot(kind='bar', ax=axes[1, 2], color=['red', 'green'])
axes[1, 2].set_title('Ortalama Kampanya Sayisi x Yas Grubu', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Yas Grubu')
axes[1, 2].set_ylabel('Ortalama Kampanya Sayisi')
axes[1, 2].legend(['No', 'Yes'])
axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('cross_analysis_campaign_demo.png', dpi=300, bbox_inches='tight')
print("\n✓ Grafik kaydedildi: cross_analysis_campaign_demo.png")
plt.close()

print("\n" + "="*80 + "\n")

# ============================================================================
# 4. EKONOMİK GÖSTERGELER x MÜŞTERİ PROFİLİ
# ============================================================================
print("\n4. EKONOMİK GÖSTERGELER x MÜŞTERİ PROFİLİ")
print("-" * 80)

# Euribor grupları oluştur
data['euribor_group'] = pd.cut(data['euribor3m'], 
                               bins=[-1, 1, 2, 3, 6],
                               labels=['Cok Dusuk (<1)', 'Dusuk (1-2)', 'Orta (2-3)', 'Yuksek (>3)'])

# Euribor x Yaş grubu
euribor_age = pd.crosstab([data['euribor_group'], data['age_group']], 
                          data['y'], normalize='index') * 100

print("\nEuribor Seviyesi ve Yaş Grubuna Göre Abonelik Oranları (%):")
print(euribor_age['yes'].unstack().round(2))

# Euribor x Meslek
euribor_job = pd.crosstab([data[data['job'].isin(top_jobs[:5])]['euribor_group'], 
                           data[data['job'].isin(top_jobs[:5])]['job']], 
                          data[data['job'].isin(top_jobs[:5])]['y'], 
                          normalize='index') * 100

print("\nEuribor Seviyesi ve Mesleğe Göre Abonelik Oranları (%):")
print(euribor_job['yes'].unstack().round(2))

# Ekonomik göstergeler ve demografik özellikler korelasyonu
print("\n📊 Ekonomik Göstergeler - Demografik Özellikler İlişkisi:")
print(f"Yaş - Euribor korelasyonu: {data['age'].corr(data['euribor3m']):.3f}")
print(f"Yaş - Tüketici Güven Endeksi korelasyonu: {data['age'].corr(data['cons.conf.idx']):.3f}")
print(f"Duration - Euribor korelasyonu: {data['duration'].corr(data['euribor3m']):.3f}")

# Görselleştirme
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 4.1 Euribor x Yaş heatmap
euribor_age_pivot = euribor_age['yes'].unstack()
sns.heatmap(euribor_age_pivot, annot=True, fmt='.1f', cmap='coolwarm', 
            ax=axes[0, 0], cbar_kws={'label': 'Abonelik Orani (%)'})
axes[0, 0].set_title('Euribor Seviyesi x Yas Grubu', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Yas Grubu')
axes[0, 0].set_ylabel('Euribor Seviyesi')

# 4.2 Euribor x Meslek
euribor_job_pivot = euribor_job['yes'].unstack()
euribor_job_pivot.plot(kind='bar', ax=axes[0, 1], colormap='Spectral')
axes[0, 1].set_title('Euribor Seviyesi x Meslek', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Euribor Seviyesi')
axes[0, 1].set_ylabel('Abonelik Orani (%)')
axes[0, 1].legend(title='Meslek', bbox_to_anchor=(1.05, 1))
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')

# 4.3 Scatter: Yaş vs Euribor (colored by y)
sample = data.sample(n=min(3000, len(data)), random_state=42)
colors = ['red' if y == 'no' else 'green' for y in sample['y']]
axes[1, 0].scatter(sample['age'], sample['euribor3m'], c=colors, alpha=0.3, s=10)
axes[1, 0].set_xlabel('Yas')
axes[1, 0].set_ylabel('Euribor 3m')
axes[1, 0].set_title('Yas vs Euribor (Abonelik Durumuna Gore)', fontsize=12, fontweight='bold')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', alpha=0.5, label='Yes'),
                   Patch(facecolor='red', alpha=0.5, label='No')]
axes[1, 0].legend(handles=legend_elements)

# 4.4 Ekonomik göstergeler box plot (yes vs no)
economic_data = []
for var in ['euribor3m', 'cons.conf.idx', 'emp.var.rate']:
    for outcome in ['no', 'yes']:
        subset = data[data['y'] == outcome][var]
        economic_data.append({'Variable': var, 'Outcome': outcome, 'Value': subset.mean()})

economic_df = pd.DataFrame(economic_data)
economic_pivot = economic_df.pivot(index='Variable', columns='Outcome', values='Value')
economic_pivot.plot(kind='barh', ax=axes[1, 1], color=['red', 'green'])
axes[1, 1].set_title('Ekonomik Gostergeler Ortalamasi (Abonelik Durumuna Gore)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Ortalama Deger')
axes[1, 1].set_ylabel('Ekonomik Gosterge')
axes[1, 1].legend(['No', 'Yes'])

plt.tight_layout()
plt.savefig('cross_analysis_economic_profile.png', dpi=300, bbox_inches='tight')
print("\n✓ Grafik kaydedildi: cross_analysis_economic_profile.png")
plt.close()

print("\n" + "="*80 + "\n")
print("✓ TÜM ÇAPRAZ ANALİZLER TAMAMLANDI!")
print("\nOluşturulan Grafikler:")
print("  1. cross_analysis_marital_age.png")
print("  2. cross_analysis_job_edu_age.png")
print("  3. cross_analysis_campaign_demo.png")
print("  4. cross_analysis_economic_profile.png")
print("\n" + "="*80)
