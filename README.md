# Kaggle Playground S6E10: Predicting Airline Satisfaction

Hedef `satisfaction` (True/False), gönderim **olasılık**, metrik **ROC AUC**. Bu repo uçtan uca çözümü içerir:
veri inceleme, sabit 5 katlı CV, özellik grupları, LightGBM / XGBoost / CatBoost / MLP / seyrek lojistik
regresyon, Optuna ile ayar ve OOF üzerinde harman. Tüm deneyler `experiments.md`'de.

<!-- RESULTS -->

## Veri

| | Satır | Not |
|---|---|---|
| train | 699.635 | %44,4 memnun |
| test | 299.844 | |
| orijinal (`arseniyshutko/binary-aviation-satisfaction-129k`) | 129.880 | %43,5 memnun |

- 22 özellik: 4 kategorik (Gender, Customer Type, Type of Travel, Class), 13 hizmet puanı (0–5), Age,
  Flight Distance, Departure / Arrival Delay. Eksik değer yalnız Arrival Delay'de (train 292, test 130).
- **Train ile test aynı dağılımdan:** düşmanca doğrulama AUC'si 0.500.
- **Orijinal veri belirgin biçimde farklı:** train ile orijinal arasında düşmanca AUC 0.834. Farkı yaratanlar
  gecikmeler (train'de %91'i 0, orijinalde %56) ve mesafe.
- **Flight Distance bir "rota kimliği" gibi davranıyor.** 200'den fazla satırı olan mesafe değerlerinde memnuniyet
  oranının standart sapması 0.21; binom gürültüsü yalnız 0.02 olurdu. Class / Type of Travel / Customer Type
  hücresi kontrol edildikten sonra bile 0.086 kalıyor. Bu oranlar train ile orijinal veri arasında 0.89
  korelasyonlu. Mesafe değerlerinin %99,999'u orijinal veride de var.
- Puanlarda 0 ("not applicable") az ama anlamlı: 3 ya da daha fazla puanı 0 olan yolcuların %86–94'ü memnun.

**Kurallar ve dış veri:** yarışma kuralları (2.6) herkese açık, ücretsiz dış veriye izin veriyor. Orijinal veri
yalnız (a) her eğitim katına ek satır (`is_orig` bayrağıyla), (b) orijinal veride eğitilmiş bir modelin tahmini
(`opred`) ve (c) mesafe başına profil olarak denendi. Test satırlarını orijinalle eşleştirip etiket kopyalanmadı
(zaten yalnız 4 test satırı orijinalle birebir aynı).

## Validation

- `StratifiedKFold(5, shuffle=True, random_state=42)`; her modelde aynı katlar (`src/common.py`).
  `seed=` yalnız model tohumunu değiştirir.
- Her deney `oof/<etiket>.npy` ve `preds/<etiket>.npy` yazar ve `experiments.md`'ye bir satır ekler
  (kat ortalaması ± std ve havuzlanmış OOF AUC).
- Hedef kodlama yalnız kat içinde: sklearn `TargetEncoder(cv=5)` her dış katın eğitim satırlarında (orijinal satırlar
  dahil) fit edilir, doğrulama ve test o katın kodlayıcısıyla dönüşür. Kodlamalar kolon × kat başına diske önbelleklenir
  (`src/te_cache.py`).
- Erken durdurma doğrulama katının AUC'siyle.
- Gürültü düzeyi: aynı özellik seti, üç model tohumu → 0.95986 / 0.95993 / 0.95997 (std ≈ 0.00005). 0.0001'in
  altındaki farkları gürültü saydım.
- Harmanlar iç içe ölçüldü: ağırlıklar OOF'un 4/5'inde seçilir, kalan 1/5'te puanlanır (`src/blend.py`).

## Özellikler (LightGBM, lr 0.1, 5 kat CV)

Her satır bir öncekinin üstüne eklenir. Kat ortalaması; tek tohum (gürültü std ≈ 0.00005).

| Adım | CV AUC | Fark |
|---|---|---|
| Ham kolonlar, LightGBM varsayılanları (100 ağaç) | 0.95780 | |
| Ham kolonlar, 63 yaprak + erken durdurma (`std`) | 0.95879 | +0.00099 |
| + `te1`: her kolonun kat içi hedef kodlaması (her kolon kategori gibi) | 0.95997 | +0.00118 |
| + `tefd`: Flight Distance × diğer her kolonun hedef kodlaması | 0.96018 | +0.00021 |
| + `fdprof`: mesafe değeri başına etiketsiz profil (Business payı, yaş, puan ortalamaları; train+test) | 0.96020 | |
| + `cnt` (değer sayımları) + `inter` (Customer Type × Type of Travel × Class) | 0.96021 | |
| + orijinal satırlar her eğitim katında (`orig=1`, `is_orig` bayrağı) → **FS1** | 0.96052 | +0.00031 |
| + `te2`: 20 düşük kardinaliteli kolonun bütün ikililerinin hedef kodlaması → **FS2** | 0.96082 | +0.00030 |
| + `cnt2` (ikili sayımları) + `opred` (orijinalde eğitilmiş modelin tahmini) → **FS3** | 0.96089 | +0.00007 |

Tek başına denenip tutulmayanlar (ham kolonlar üstüne, referans 0.95879):

| Grup | CV | |
|---|---|---|
| `afill`: eksik Arrival Delay'i Departure Delay ile doldurma | 0.95876 | gürültü |
| `delay`: gecikme toplamı / farkı / max / log / oranı, bayraklar | 0.95870 | gürültü |
| `rnan`: puan 0 → NaN + sıfır sayısı | 0.95880 | gürültü |
| `zflag`: her puan için 0 bayrağı | 0.95867 | gürültü |
| `agg`: puan ortalaması / min / max / std / toplam / 5 ve 1 sayıları | 0.95843 | kötü |
| `grp`: dijital / kabin / hizmet grup ortalamaları ve farkları | 0.95846 | kötü |
| `rcat`: puanların kategorik kopyası | 0.95866 | gürültü |
| `tecat`: CT × ToT × Class ve her puanla üçlülerin hedef kodlaması | 0.95876 | gürültü |
| `omean`: orijinal veride değer başına memnuniyet oranı | 0.95979 (+0.001) | `te1` varken 0 |
| `ofd`, `omean` FS1 üstüne | 0.96047 / 0.96049 | 0 |
| `opred` FS2 üstüne | 0.96082 | 0 |

Ağaç modelleri ham puan ve gecikmelerdeki etkileşimleri zaten yakalıyor; elle türetilen toplamlar ve gruplar
yalnız gürültü ekledi. Kazanç, değerlerin **kimliğini** (özellikle mesafe = rota) hedef kodlamayla ve
ikililerle vermekten ve orijinal veriden geldi.

## Modeller

Tüm modeller aynı 5 katta, orijinal satırlar eğitim katlarında. Süreler 4 çekirdekli CPU'da.

| Model | Özellikler | Ayar | CV (OOF AUC) | Süre |
|---|---|---|---|---|
| LightGBM | FS3 (473 kolon) | `t1`, lr 0.03 | **0.96130** (ikinci tohum 0.96130) | 17–21 dk |
| XGBoost | FS3 | `t1`, lr 0.03 | 0.96129 | 50 dk |
| LightGBM | FS2 (278 kolon) | `t1`, lr 0.03, tohum 7 | 0.96122 | 13 dk |
| XGBoost | FS2 | `t1`, lr 0.03 | 0.96117 | 37 dk |
| CatBoost | FS1 (88 kolon) | derinlik 6, lr 0.05 | 0.96065 | 62 dk |
| MLP | ham 21 kolon: her biri gömme + sayısal kopyalar | 12 epoch, 3 tohum ortalaması | 0.95992 (tek tohum 0.9590–0.9591) | 12 dk / tohum |
| MLP + TE girdileri | + te1, tefd, te2 (logit, standart) | 12 epoch | 0.95980 | 15 dk |
| Seyrek lojistik regresyon | her değer + 190 kolon ikilisi + FD × kategorik, one-hot | C = 0.1 | 0.95802 | 5 dk |

**Ayar (Optuna, `src/tune.py`).** FS1 üzerinde, 5 katın ilk 2'sinde, lr 0.1 ile; LightGBM 30, XGBoost 18 deneme.

| | std ayar (2 kat) | en iyi deneme (2 kat) | 5 kat, lr 0.1 |
|---|---|---|---|
| LightGBM `t1`: 112 yaprak, min_child_samples 9, subsample 0.97, colsample 0.46, λ 18.5, max_bin 511 | 0.96009 | 0.96039 | FS1 0.96052 → 0.96074, FS2 0.96084 → 0.96098 |
| XGBoost `t1`: derinlik 10, min_child_weight 83, subsample 0.90, colsample 0.44, α 0.55, max_bin 1024 | 0.95985 | 0.96032 | FS1 0.96028 → 0.96053 |

Öğrenme oranı 0.1 → 0.03: LightGBM FS3 0.96089 (std) → 0.96130 (t1); XGBoost FS2 0.96053 (FS1, lr 0.1) → 0.96117.
CatBoost lr 0.1 → 0.05: 0.96054 → 0.96065.

İşe yaramayanlar (model düzeyi):

- Orijinal satır ağırlığı 0.5 ya da 2 (FS1 t1: 0.96074 → 0.96063 / 0.96065).
- `te3` (en güçlü 8 kolonun 56 üçlüsü + 3 FD üçlüsü): FS2 t1 lr 0.1, 0.96098 → 0.96105, gürültü sınırında.
- CatBoost'ta her kolonu kategorik vermek (CTR birleşimleri): 4 çekirdekte bir kat 25 dakikada bitmedi, durduruldu.
- MLP'ye hedef kodlamaları girdi olarak vermek tek başına daha iyi (0.9591 → 0.9598) ama harmanda ağırlık
  almadı (0.003): GBDT'lere fazla benziyor. Harmana katkıyı ham girdili MLP veriyor.

## Ensemble

<!-- ENSEMBLE -->

## Gönderimler

<!-- SUBMISSIONS -->

## Çalıştırma

```
pip install kaggle lightgbm xgboost catboost optuna scikit-learn pandas torch
kaggle competitions download -c playground-series-s6e10 -p data && unzip data/playground-series-s6e10.zip -d data
kaggle datasets download arseniyshutko/binary-aviation-satisfaction-129k -p data/orig --unzip
python src/adversarial.py
python src/train.py lgbm base,te1,tefd,fdprof,cnt,inter,te2 orig=1 preset=t1 lr=0.03
python src/train.py xgb  base,te1,tefd,fdprof,cnt,inter,te2 orig=1 preset=t1 lr=0.03
python src/train.py cat  base,te1,tefd,fdprof,cnt,inter orig=1 lr=0.05
python src/glm.py pairs=all C=0.1
python src/nn.py orig=1
python src/tune.py lgbm base,te1,tefd,fdprof,cnt,inter orig=1 trials=30 lr=0.1 nfold=2
python src/blend.py <çıktı adı> <etiket> <etiket> ... method=all
```

| Dosya | İş |
|---|---|
| `src/common.py` | sabitler, veri yükleme, sabit katlar, OOF kaydı |
| `src/features.py` | etiketsiz özellik grupları ve hedef kodlama anahtarları |
| `src/te_cache.py` | kat içi hedef kodlama, kolon × kat önbelleği |
| `src/train.py` | LightGBM / XGBoost / CatBoost K katlı CV; kat checkpoint'i, tamamlananı atlama |
| `src/glm.py` | one-hot + ikililer üstünde seyrek lojistik regresyon |
| `src/nn.py` | kolon başına gömmeli MLP (PyTorch, CPU) |
| `src/tune.py` | Optuna (sqlite'ta kaldığı yerden devam eder) |
| `src/blend.py` | hill climbing / sıra ortalaması / olasılık ortalaması / lojistik istifleme, iç içe CV |
| `src/submit_check.py` | gönderimi sample_submission sırasıyla yazar ve biçimini denetler |
| `src/adversarial.py` | düşmanca doğrulama |
| `scripts/*.sh` | arka planda çalıştırılan deney kuyrukları |
