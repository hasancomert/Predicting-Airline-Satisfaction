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

<!-- FEATURES -->

## Modeller

<!-- MODELS -->

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
