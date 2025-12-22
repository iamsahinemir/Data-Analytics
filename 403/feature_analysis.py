import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("Veri seti yükleniyor...")
data = pd.read_csv('dataset/bank-additional-full.csv', sep=';')

data = data.drop_duplicates()
print(f"Toplam kayıt sayısı: {len(data)}")
print(f"Hedef değişken dağılımı:\n{data['y'].value_counts()}")
print("\n" + "="*80 + "\n")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("1. YAŞ ANALİZİ")
print("-" * 80)

data['age_group'] = pd.cut(data['age'], 
                            bins=[0, 25, 35, 45, 55, 65, 100],
                            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
age_analysis = pd.crosstab(data['age_group'], data['y'], normalize='index') * 100
print("\nYaş Gruplarına Göre Term Deposit Abonelik Oranları (%):")
print(age_analysis.round(2))

print("\nYaş İstatistikleri:")
print(f"Ortalama yaş: {data['age'].mean():.2f}")
print(f"Medyan yaş: {data['age'].median():.2f}")
print(f"En genç: {data['age'].min()}, En yaşlı: {data['age'].max()}")

print("\nHedef Değişkene Göre Ortalama Yaş:")
print(data.groupby('y')['age'].agg(['mean', 'median', 'std']).round(2))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].hist([data[data['y']=='yes']['age'], data[data['y']=='no']['age']], 
                bins=30, label=['Yes', 'No'], alpha=0.7, color=['green', 'red'])
axes[0, 0].set_xlabel('Yas')
axes[0, 0].set_ylabel('Frekans')
axes[0, 0].set_title('Yas Dagilimi - Term Deposit Aboneligi')
axes[0, 0].legend()

age_analysis['yes'].plot(kind='bar', ax=axes[0, 1], color='steelblue')
axes[0, 1].set_xlabel('Yas Grubu')
axes[0, 1].set_ylabel('Abonelik Orani (%)')
axes[0, 1].set_title('Yas Gruplarina Gore Abonelik Orani')
axes[0, 1].set_xticklabels(age_analysis.index, rotation=45)

data.boxplot(column='age', by='y', ax=axes[1, 0])
axes[1, 0].set_xlabel('Term Deposit')
axes[1, 0].set_ylabel('Yas')
axes[1, 0].set_title('Yas Dagilimi - Box Plot')
plt.sca(axes[1, 0])
plt.xticks([1, 2], ['No', 'Yes'])

age_counts = data['age_group'].value_counts().sort_index()
axes[1, 1].bar(range(len(age_counts)), age_counts.values, color='coral')
axes[1, 1].set_xlabel('Yas Grubu')
axes[1, 1].set_ylabel('Kayit Sayisi')
axes[1, 1].set_title('Yas Grubu Dagilimi')
axes[1, 1].set_xticks(range(len(age_counts)))
axes[1, 1].set_xticklabels(age_counts.index, rotation=45)

plt.tight_layout()
plt.savefig('age_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Yaş analizi grafiği kaydedildi: age_analysis.png")
plt.close()

print("\n" + "="*80 + "\n")

print("2. MESLEK ANALİZİ")
print("-" * 80)

job_analysis = pd.crosstab(data['job'], data['y'], normalize='index') * 100
job_analysis = job_analysis.sort_values('yes', ascending=False)
print("\nMesleklere Göre Term Deposit Abonelik Oranları (%):")
print(job_analysis.round(2))

job_counts = data['job'].value_counts()
print("\nEn Yaygın Meslekler:")
print(job_counts.head(10))

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

job_analysis['yes'].plot(kind='barh', ax=axes[0, 0], color='teal')
axes[0, 0].set_xlabel('Abonelik Orani (%)')
axes[0, 0].set_ylabel('Meslek')
axes[0, 0].set_title('Mesleklere Gore Abonelik Orani')

job_counts.plot(kind='barh', ax=axes[0, 1], color='orange')
axes[0, 1].set_xlabel('Kayit Sayisi')
axes[0, 1].set_ylabel('Meslek')
axes[0, 1].set_title('Meslek Dagilimi')

top_jobs = job_analysis.nlargest(5, 'yes')
bottom_jobs = job_analysis.nsmallest(5, 'yes')
combined = pd.concat([top_jobs, bottom_jobs])
combined['yes'].plot(kind='bar', ax=axes[1, 0], color=['green']*5 + ['red']*5)
axes[1, 0].set_xlabel('Meslek')
axes[1, 0].set_ylabel('Abonelik Orani (%)')
axes[1, 0].set_title('En Yuksek ve En Dusuk Abonelik Oranli Meslekler')
axes[1, 0].set_xticklabels(combined.index, rotation=45, ha='right')

job_age_pivot = pd.crosstab(data['job'], data['age_group'])
job_age_pivot_top = job_age_pivot.loc[job_counts.head(8).index]
job_age_pivot_top.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='viridis')
axes[1, 1].set_xlabel('Meslek')
axes[1, 1].set_ylabel('Kayit Sayisi')
axes[1, 1].set_title('Meslek ve Yas Grubu Iliskisi (En Yaygin 8 Meslek)')
axes[1, 1].set_xticklabels(job_age_pivot_top.index, rotation=45, ha='right')
axes[1, 1].legend(title='Yas Grubu', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('job_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Meslek analizi grafiği kaydedildi: job_analysis.png")
plt.close()

print("\n" + "="*80 + "\n")

print("3. EĞİTİM SEVİYESİ ANALİZİ")
print("-" * 80)

education_analysis = pd.crosstab(data['education'], data['y'], normalize='index') * 100
education_analysis = education_analysis.sort_values('yes', ascending=False)
print("\nEğitim Seviyesine Göre Term Deposit Abonelik Oranları (%):")
print(education_analysis.round(2))

education_counts = data['education'].value_counts()
print("\nEğitim Seviyesi Dağılımı:")
print(education_counts)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

education_analysis['yes'].plot(kind='bar', ax=axes[0, 0], color='purple')
axes[0, 0].set_xlabel('Egitim Seviyesi')
axes[0, 0].set_ylabel('Abonelik Orani (%)')
axes[0, 0].set_title('Egitim Seviyesine Gore Abonelik Orani')
axes[0, 0].set_xticklabels(education_analysis.index, rotation=45, ha='right')

axes[0, 1].pie(education_counts.values, labels=education_counts.index, autopct='%1.1f%%', startangle=90)
axes[0, 1].set_title('Egitim Seviyesi Dagilimi')

edu_age = data.groupby('education')['age'].mean().sort_values(ascending=False)
edu_age.plot(kind='barh', ax=axes[1, 0], color='indianred')
axes[1, 0].set_xlabel('Ortalama Yas')
axes[1, 0].set_ylabel('Egitim Seviyesi')
axes[1, 0].set_title('Egitim Seviyesine Gore Ortalama Yas')

top_5_jobs = job_counts.head(5).index
edu_job_data = data[data['job'].isin(top_5_jobs)]
edu_job_pivot = pd.crosstab(edu_job_data['education'], edu_job_data['job'], normalize='index') * 100
edu_job_pivot.plot(kind='bar', ax=axes[1, 1], colormap='Set3')
axes[1, 1].set_xlabel('Egitim Seviyesi')
axes[1, 1].set_ylabel('Oran (%)')
axes[1, 1].set_title('Egitim Seviyesi ve Meslek Iliskisi (Top 5 Meslek)')
axes[1, 1].set_xticklabels(edu_job_pivot.index, rotation=45, ha='right')
axes[1, 1].legend(title='Meslek', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('education_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Eğitim seviyesi analizi grafiği kaydedildi: education_analysis.png")
plt.close()

print("\n" + "="*80 + "\n")

print("4. MEDENİ DURUM ANALİZİ")
print("-" * 80)

marital_analysis = pd.crosstab(data['marital'], data['y'], normalize='index') * 100
marital_analysis = marital_analysis.sort_values('yes', ascending=False)
print("\nMedeni Duruma Göre Term Deposit Abonelik Oranları (%):")
print(marital_analysis.round(2))

marital_counts = data['marital'].value_counts()
print("\nMedeni Durum Dağılımı:")
print(marital_counts)

print("\nMedeni Duruma Göre Ortalama Yaş:")
print(data.groupby('marital')['age'].agg(['mean', 'median', 'std']).round(2))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

marital_analysis['yes'].plot(kind='bar', ax=axes[0, 0], color='darkgreen')
axes[0, 0].set_xlabel('Medeni Durum')
axes[0, 0].set_ylabel('Abonelik Orani (%)')
axes[0, 0].set_title('Medeni Duruma Gore Abonelik Orani')
axes[0, 0].set_xticklabels(marital_analysis.index, rotation=0)

marital_counts.plot(kind='bar', ax=axes[0, 1], color='skyblue')
axes[0, 1].set_xlabel('Medeni Durum')
axes[0, 1].set_ylabel('Kayit Sayisi')
axes[0, 1].set_title('Medeni Durum Dagilimi')
axes[0, 1].set_xticklabels(marital_counts.index, rotation=0)

for marital_status in data['marital'].unique():
    subset = data[data['marital'] == marital_status]['age']
    axes[1, 0].hist(subset, alpha=0.5, label=marital_status, bins=30)
axes[1, 0].set_xlabel('Yas')
axes[1, 0].set_ylabel('Frekans')
axes[1, 0].set_title('Medeni Durum ve Yas Dagilimi')
axes[1, 0].legend()

marital_age_y = pd.crosstab([data['marital'], data['age_group']], data['y'], normalize='index') * 100
marital_age_y['yes'].unstack(level=0).plot(kind='bar', ax=axes[1, 1], colormap='Paired')
axes[1, 1].set_xlabel('Yas Grubu')
axes[1, 1].set_ylabel('Abonelik Orani (%)')
axes[1, 1].set_title('Medeni Durum ve Yas Grubuna Gore Abonelik Orani')
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)
axes[1, 1].legend(title='Medeni Durum')

plt.tight_layout()
plt.savefig('marital_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Medeni durum analizi grafiği kaydedildi: marital_analysis.png")
plt.close()

print("\n" + "="*80 + "\n")

print("5. KAMPANYA ÖZELLİKLERİ ANALİZİ")
print("-" * 80)

print("\nArama Süresi (Duration) İstatistikleri:")
print(data.groupby('y')['duration'].agg(['mean', 'median', 'std', 'min', 'max']).round(2))

print("\nKampanya İletişim Sayısı İstatistikleri:")
print(data.groupby('y')['campaign'].agg(['mean', 'median', 'std', 'min', 'max']).round(2))

poutcome_analysis = pd.crosstab(data['poutcome'], data['y'], normalize='index') * 100
poutcome_analysis = poutcome_analysis.sort_values('yes', ascending=False)
print("\nÖnceki Kampanya Sonucuna Göre Abonelik Oranları (%):")
print(poutcome_analysis.round(2))

contact_analysis = pd.crosstab(data['contact'], data['y'], normalize='index') * 100
print("\nİletişim Türüne Göre Abonelik Oranları (%):")
print(contact_analysis.round(2))

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

axes[0, 0].hist([data[data['y']=='yes']['duration'], data[data['y']=='no']['duration']], 
                bins=50, label=['Yes', 'No'], alpha=0.7, color=['green', 'red'], range=(0, 2000))
axes[0, 0].set_xlabel('Arama Suresi (saniye)')
axes[0, 0].set_ylabel('Frekans')
axes[0, 0].set_title('Arama Suresi Dagilimi')
axes[0, 0].legend()

campaign_groups = data.groupby(['campaign', 'y']).size().unstack(fill_value=0)
campaign_groups_pct = campaign_groups.div(campaign_groups.sum(axis=1), axis=0) * 100
campaign_groups_pct.iloc[:10]['yes'].plot(kind='bar', ax=axes[0, 1], color='steelblue')
axes[0, 1].set_xlabel('Kampanya Iletisim Sayisi')
axes[0, 1].set_ylabel('Abonelik Orani (%)')
axes[0, 1].set_title('Kampanya Iletisim Sayisina Gore Abonelik Orani')
axes[0, 1].set_xticklabels(range(1, 11), rotation=0)

poutcome_analysis['yes'].plot(kind='bar', ax=axes[0, 2], color='darkblue')
axes[0, 2].set_xlabel('Onceki Kampanya Sonucu')
axes[0, 2].set_ylabel('Abonelik Orani (%)')
axes[0, 2].set_title('Onceki Kampanya Sonucuna Gore Abonelik Orani')
axes[0, 2].set_xticklabels(poutcome_analysis.index, rotation=45, ha='right')

contact_analysis['yes'].plot(kind='bar', ax=axes[1, 0], color='teal')
axes[1, 0].set_xlabel('Iletisim Turu')
axes[1, 0].set_ylabel('Abonelik Orani (%)')
axes[1, 0].set_title('Iletisim Turune Gore Abonelik Orani')
axes[1, 0].set_xticklabels(contact_analysis.index, rotation=0)

month_analysis = pd.crosstab(data['month'], data['y'], normalize='index') * 100
month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month_analysis = month_analysis.reindex([m for m in month_order if m in month_analysis.index])
month_analysis['yes'].plot(kind='bar', ax=axes[1, 1], color='coral')
axes[1, 1].set_xlabel('Ay')
axes[1, 1].set_ylabel('Abonelik Orani (%)')
axes[1, 1].set_title('Aya Gore Abonelik Orani')
axes[1, 1].set_xticklabels(month_analysis.index, rotation=45)

sample_data = data.sample(n=min(5000, len(data)), random_state=42)
colors = ['green' if y == 'yes' else 'red' for y in sample_data['y']]
axes[1, 2].scatter(sample_data['campaign'], sample_data['duration'], 
                   c=colors, alpha=0.3, s=10)
axes[1, 2].set_xlabel('Kampanya Iletisim Sayisi')
axes[1, 2].set_ylabel('Arama Suresi (saniye)')
axes[1, 2].set_title('Kampanya Sayisi vs Arama Suresi')
axes[1, 2].set_xlim(0, 20)
axes[1, 2].set_ylim(0, 2000)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', alpha=0.5, label='Yes'),
                   Patch(facecolor='red', alpha=0.5, label='No')]
axes[1, 2].legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('campaign_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Kampanya özellikleri analizi grafiği kaydedildi: campaign_analysis.png")
plt.close()

print("\n" + "="*80 + "\n")

print("6. EKONOMİK GÖSTERGELER ANALİZİ")
print("-" * 80)

economic_vars = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

print("\nEkonomik Göstergelerin Hedef Değişkene Göre Ortalamaları:")
print(data.groupby('y')[economic_vars].mean().round(2))

print("\nEkonomik Göstergeler Arası Korelasyon:")
correlation_matrix = data[economic_vars].corr()
print(correlation_matrix.round(2))

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            ax=axes[0, 0], fmt='.2f', square=True)
axes[0, 0].set_title('Ekonomik Gostergeler Korelasyon Matrisi')

for idx, var in enumerate(economic_vars[:5]):
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    data.boxplot(column=var, by='y', ax=axes[row, col])
    axes[row, col].set_xlabel('Term Deposit')
    axes[row, col].set_ylabel(var)
    axes[row, col].set_title(f'{var} Dagilimi')
    plt.sca(axes[row, col])
    plt.xticks([1, 2], ['No', 'Yes'])

plt.tight_layout()
plt.savefig('economic_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Ekonomik göstergeler analizi grafiği kaydedildi: economic_analysis.png")
plt.close()

print("\n" + "="*80 + "\n")
print("✓ TÜM ANALİZLER TAMAMLANDI!")
print("\nOluşturulan Grafikler:")
print("  1. age_analysis.png")
print("  2. job_analysis.png")
print("  3. education_analysis.png")
print("  4. marital_analysis.png")
print("  5. campaign_analysis.png")
print("  6. economic_analysis.png")
