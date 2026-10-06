# Kaggle Playground S6E10: Predicting Airline Satisfaction

Hedef `satisfaction` (True/False), gönderim **olasılık**, metrik **ROC AUC**. Bu repo uçtan uca çözümü içerir:
veri inceleme, sabit 5 katlı CV, özellik grupları, LightGBM / XGBoost / CatBoost / MLP / seyrek lojistik
regresyon, Optuna ile ayar ve OOF üzerinde harman. Tüm deneyler `experiments.md`'de.

**Sonuç:** 8 model ailesinin lojistik istiflemesi, 5 katlı CV AUC **0.96160** (iç içe). Gönderilen en iyi:
public **0.96087** (2. gönderim, CV 0.96147). Baseline LightGBM (varsayılanlar) 0.95780'di.

Ne işe yaradı (büyükten küçüğe):

1. Değerlerin kimliğini kat içi hedef kodlamayla vermek (`te1` +0.0012). Özellikle Flight Distance bir rota
   kimliği gibi davranıyor; mesafe × diğer kolon kodlamaları (+0.0002) ve kolon ikilileri (`te2`, +0.0003).
2. Orijinal veriyi her eğitim katına ek satır olarak koymak (+0.0002–0.0003).
3. Ayar ve düşük öğrenme oranı: LightGBM 0.96089 (std, lr 0.1) → 0.96130 (t1, lr 0.03).
4. Farklı ailelerden harman: tek en iyi model 0.96130 → 0.96160. En değerli üyeler hedef kodlamasız LightGBM
   ve 7 tohumlu ham girdili MLP.

İşe yaramayanlar: elle türetilmiş gecikme / puan özetleri, puan grupları, 0 bayrakları, orijinal veride değer
başına oranlar (TE varken), CatBoost'ta her kolonu kategorik vermek, orijinal satır ağırlığını değiştirmek, TE
girdili MLP'yi harmana katmak, üçlü hedef kodlamalar (gürültü sınırında).

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
| RealMLP (pytabkit) | ham kolonlar + her sayısal kolonun kategorik ikizi | açık notebook ayarları, 3 epoch, n_ens 8 | 0.96076 | 19 dk |
| RealMLP | aynı, 6 epoch, tohum 1 | | 0.96100 | 41 dk |
| RealMLP v4 | + rota profili, sayımlar, orijinal-model logit'i, kat içi TE (te1, tefd) | 6 epoch | 0.96114 | 64 dk |

**Ayar (Optuna, `src/tune.py`).** FS1 üzerinde, 5 katın ilk 2'sinde, lr 0.1 ile; LightGBM 30, XGBoost 18 deneme.

| | std ayar (2 kat) | en iyi deneme (2 kat) | 5 kat, lr 0.1 |
|---|---|---|---|
| LightGBM `t1`: 112 yaprak, min_child_samples 9, subsample 0.97, colsample 0.46, λ 18.5, max_bin 511 | 0.96009 | 0.96039 | FS1 0.96052 → 0.96074, FS2 0.96084 → 0.96098 |
| XGBoost `t1`: derinlik 10, min_child_weight 83, subsample 0.90, colsample 0.44, α 0.55, max_bin 1024 | 0.95985 | 0.96032 | FS1 0.96028 → 0.96053 |

Öğrenme oranı 0.1 → 0.03: LightGBM FS3 0.96089 (std) → 0.96130 (t1); XGBoost FS2 0.96053 (FS1, lr 0.1) → 0.96117.
CatBoost lr 0.1 → 0.05: 0.96054 → 0.96065.

RealMLP ayarları ve "her sayısal kolona kategorik ikiz" fikri yekenot'un açık *PS|S6|E10: RealMLP · PyTabKit*
notebook'undan, v4 özellik görünümü goodpjw2008'in *Route-ID FE + OG-Model Stack* notebook'undan esinlenildi;
eğitim bizim katlarımızda ve bizim özelliklerimizle (`src/realmlp.py`). CPU'da bir kat 4–13 dakika.

İşe yaramayanlar (model düzeyi):

- Orijinal satır ağırlığı 0.5 ya da 2 (FS1 t1: 0.96074 → 0.96063 / 0.96065).
- `te3` (en güçlü 8 kolonun 56 üçlüsü + 3 FD üçlüsü): FS2 t1 lr 0.1, 0.96098 → 0.96105, gürültü sınırında.
- CatBoost'ta her kolonu kategorik vermek (CTR birleşimleri): 4 çekirdekte bir kat 25 dakikada bitmedi, durduruldu.
- MLP'ye hedef kodlamaları girdi olarak vermek tek başına daha iyi (0.9591 → 0.9598) ama harmanda ağırlık
  almadı (0.003): GBDT'lere fazla benziyor. Harmana katkıyı ham girdili MLP veriyor.

## 10 kat (s05)

s04'e kadar her şey 5 katlıydı ve harman 0.9616'da doyuma ulaştı. s05 için seçilen aileler aynı
`StratifiedKFold(10, shuffle=True, random_state=42)` ile yeniden eğitildi (her model verinin %90'ını görür).
LightGBM ve CatBoost bu makinede (CPU), XGBoost / RealMLP / MLP özel Kaggle not defterlerinde GPU'da
(`kaggle/gpu_kernel.py`: `src/` paketlenir, komutlar çalışır, `.npy` çıktıları geri indirilir; hedef kodlama orada
diske önbelleklenmeden aynı kodla hesaplanır, değerler birebir aynıdır).

| Üye | 5 kat | 10 kat |
|---|---|---|
| LightGBM FS3 t1 (2 tohum) | 0.96130 / 0.96130 | 0.96144 / 0.96146 |
| XGBoost FS3 t1 (2 tohum, GPU) | 0.96129 | 0.96148 / 0.96145 |
| Hedef kodlamasız LightGBM t1 / t2 | 0.96098 / 0.96095 | 0.96106 / 0.96109 |
| RealMLP v4 (3 tohum, GPU) | 0.96114 / 0.96107 | 0.96124 / 0.96125 / 0.96123 |
| RealMLP pub (3 koşu, GPU) | 0.96076 / 0.96100 | 0.96088 / 0.96110 / 0.96112 |
| MLP, tek tohum (GPU) | 0.9589–0.9591 | 0.9592–0.9594 |
| MLP, tohum ortalaması | 10 tohum 0.96022 | 15 tohum 0.96043 |

Ortalama kazanç model başına +0.0001–0.0002, önceki yarışmadaki 5 → 10 kat gözlemiyle uyumlu.

10 katta yeni bir üye de eklendi: RealMLP v4'e `te2` ikili kodlamalarını da girdi olarak vermek (3 tohum
0.96143 / 0.96144 / 0.96145; ortalaması 0.96153) en güçlü sinir ağı oldu. 10 katlı 8 aile harmanı: hill climbing
iç içe 0.96181, eşit sıra ortalaması 0.96180, lojistik istifleme 0.96180 (s04: 0.96165).

s06 / s07 turunda 10 katta denenenler:

| Üye | 10 kat CV | Harmana etkisi |
|---|---|---|
| LightGBM FS2 + `te3` (üçlü kodlamalar) + opred, 2 tohum | 0.96146 / 0.96142 | ≈0 |
| XGBoost FS3 + `te3` | 0.96144 | ≈0 |
| XGBoost FS3 3. tohum, CatBoost FS2 2. tohum | 0.96142 / 0.96105 | ≈0 |
| RealMLP v5 (v4+te2 + orijinal veride eğitilmiş RealMLP'nin logit'i), 2 tohum | 0.96145 / 0.96143 | ≈0 |
| **TabM** (pytabkit `TabM_D`, v4+te2 özellikleri, ikizsiz), 3 tohum | **0.96137 / 0.96137 / 0.96133** (ortalama 0.96142) | +0.00003, en büyük ağırlık |

Denenip tutulmayanlar: harman ağırlıklarını segment (Class, Type of Travel) başına ayrı öğrenmek (0.96184 →
0.96184); LightGBM'i `rank_xendcg` sıralama hedefiyle, rastgele 100 / 1000 satırlık gruplarda eğitmek (ilk iki katta
0.9601 / 0.9591, ikili hedef 0.9613); bütün kolonları LightGBM'e kategorik olarak da vermek (0.95995).

## Açık OOF kütüphaneleri (s08)

Yarışmanın ilk haftasında pek çok katılımcı aynı train satırları üzerinde OOF ve test tahminlerini paylaştı (çoğu
`StratifiedKFold(5, shuffle=True, random_state=42)`). Kurallar herkese açık dış veriye izin veriyor. `src/ext_pub.py`
bunları `ext/pub/` altından okuyup id'ye göre hizalar, şekil / sonluluk denetler ve OOF AUC'si makul aralık dışında
olanı (sızdırmış olabilecek, > 0.9625) reddeder. 124 üye kabul edildi, hiçbiri reddedilmedi.

| Kaynak | Üye | En iyi tek OOF |
|---|---|---|
| goodpjw2008 · *TabPFN + Route Categories* (GBDT / RealMLP / DeepFM / TabPFN-3.5 / TabICL) | 48 | 0.96134 |
| sachith7 · *stack OOF predictions* (10'u 10 katlı) | 22 | 0.96141 |
| dariushafshar · *golem OOF library* | 14 | 0.96123 |
| megayak · *OOF library* (RealMLP tohumları ortalanmış) | 11 | 0.96123 |
| mitudru · *equality blocks* | 7 | 0.96131 |
| busyaprime · *Route or distance? Both* (kratosyan'ın OOF veri seti) | 5 | 0.96122 |
| arhancanli12, thisray, megayak honest stack, wangxintong111 | 13 | 0.96109 |
| TabPFN-3.5 (samanyu1808 ×2, hemingweb) | 3 | 0.96127 |
| najiama · blend 01 | 1 | 0.96192 |

Fikir ve yükleme düzeni kratosyan'ın *Stacking every public OOF library* not defterinden; tüm emek üyelerin
yazarlarına ait. İstifleyici `src/stack.py`: logit'ler üzerinde lojistik regresyon, iç içe CV (meta katlar
`StratifiedKFold(5, seed 7)`).

| İstif | İç içe CV | Public LB |
|---|---|---|
| Yalnız kendi 10 ailemiz (s07) | 0.96184 | 0.96119 |
| Yalnız 124 açık üye (C=1) | 0.96205 | (yazarı: 0.96167) |
| Kendi + açık (C=1) | 0.96210 | |
| **Kendi + açık (C=0.001)** | **0.96213** | **0.96165** |

Kaynakları tek tek çıkarınca (C=0.001) en büyük kayıp bizim ailelerimizde: −0.00007 (goodpjw2008'in 48 üyesi
−0.00003, busyaprime −0.00002, kalanlar ≤ 0.00001). Yani kendi 10 katlı, farklı ailelerden üyelerimiz (özellikle
hedef kodlamasız LightGBM, CatBoost, TabM, RealMLP) açık havuza yeni bilgi katıyor. Farklı kat düzenlerini
karıştırmak doğrusal istiflemede sorun değil: her üyenin her satır tahmini o satırı görmemiş bir modelden.

## Ensemble

`src/blend.py` dört yöntemi aynı aday listesinde karşılaştırır ve iç içe CV'ye göre seçer: tekrar seçilebilir
hill climbing (sıra uzayında), eşit ağırlıklı sıra ortalaması, eşit ağırlıklı olasılık ortalaması ve logit(p)
üzerinde lojistik regresyon istifleme. Virgülle verilen etiketler (tohumlar) önce ortalanır.

Son aday listesi (8 aile, s03):

| Üye | Tek başına CV | Lojistik ağırlık |
|---|---|---|
| LightGBM FS3 t1 lr 0.03, 2 tohum ortalaması | 0.96130 | 0.273 |
| LightGBM, hedef kodlamasız (`base,cnt,inter,fdprof`), t1 / t2 / t2 tohum 7 ortalaması | 0.96106 | 0.234 |
| XGBoost FS3 t1 lr 0.03 | 0.96129 | 0.183 |
| MLP, 7 koşu ortalaması | 0.96017 | 0.140 |
| CatBoost FS2 lr 0.08 | 0.96096 | 0.087 |
| XGBoost FS2 t1 lr 0.03 | 0.96117 | 0.076 |
| LightGBM FS2 lr 0.03 (t1 tohum 7 + t2) | 0.9612 | −0.007 |
| Seyrek lojistik regresyon | 0.95802 | 0.000 |

| Yöntem | OOF AUC | İç içe CV |
|---|---|---|
| Lojistik istifleme | 0.96159 | **0.96160** |
| Hill climbing | 0.96161 | 0.96159 |
| Eşit sıra ortalaması | 0.96153 | 0.96153 |
| Eşit olasılık ortalaması | 0.96146 | 0.96147 |

Gözlemler:

- Harmana en çok katkıyı **farklı** üyeler yaptı, en güçlü olanlar değil. Hedef kodlamasız LightGBM tek başına
  FS3'ün 0.00024 altında ama ikinci büyük ağırlığı aldı. Ham girdili MLP (0.9602) de öyle; TE girdili MLP ise
  tek başına daha iyi olduğu hâlde ağırlık almadı.
- MLP'de tohum ortalaması güçlü: 1 → 7 koşu 0.95907 → 0.96017 (10 koşu 0.96022, harmanda fark yok). GBDT'lerde
  ikinci tohum yalnız +0.00001–0.00003.
- Lojistik istifleme ve hill climbing her seferinde eşit ortalamalardan 0.00006–0.00013 iyi; ikisi kendi
  aralarında aynı düzeyde.

## Gönderimler

| No | Dosya | İçerik | CV | İç içe CV | Public LB |
|---|---|---|---|---|---|
| 1 | `s01_blend_lr0.1_lgbmFS1-catFS1-lgbmFS2.csv` | lr 0.1 modelleri, hill climbing | 0.96113 | 0.96113 | 0.96059 |
| 2 | `s02_logitstack_lgbmFS3-xgbFS2-lr0.03_lgbm-xgb-cat-lr0.1_glm_nn.csv` | 7 model, lojistik istifleme | 0.96147 | 0.96146 | 0.96087 |
| 3 | `s03_final.csv` | 8 aile, lojistik istifleme | 0.96159 | 0.96160 | 0.96106 |
| 3-alt | `s03alt_rankavg_8families.csv` | 8 ailenin eşit sıra ortalaması | 0.96153 | 0.96153 | 0.96092 |
| 4 | `s04_hill_6fam_realmlp.csv` | RealMLP eklendi; hill climbing 6 aileyi eşit ağırlıkla seçti | 0.96165 | 0.96165 | 0.96107 |
| 5 | `s05_k10_hill_8fam.csv` | 10 kat, 8 aile (RealMLP v4+te2 dahil), hill climbing | 0.96182 | 0.96181 | 0.96114 |
| 6 | `s06_k10_logit_9fam.csv` | 10 kat, 9 aile (+te3 GBDT'ler, CatBoost 2. tohum, RealMLP v5), lojistik istifleme | 0.96181 | 0.96181 | 0.96116 |
| 7 | `s07_k10_10fam_tabm.csv` | s06 + TabM (3 tohum), lojistik istifleme | 0.96184 | 0.96184 | 0.96119 |
| 8 | `s08_stack_own10fam_pub124_C0.001.csv` | kendi 10 ailemiz + 124 açık OOF üyesi, lojistik istifleme (C=0.001) | 0.96213 | 0.96213 | 0.96165 |

Public LB CV'nin yaklaşık 0.0006 altında. 1 → 2 adımında CV +0.00033 iken public +0.00028 arttı. Public kısım
test'in küçük bir parçası; 140 bin satırlık kat std'si 0.0006 olduğundan, public skorun std'si 0.001'e yakın.
Bu yüzden kararları CV'ye göre verdim.

**Final için önerilen iki gönderim:**

1. **En iyi CV:** `s03_final.csv`, lojistik istifleme, iç içe CV 0.96160. Ağırlıklar OOF'ta öğrenildi ama
   yalnız 8 katsayı var ve iç içe ölçüm tam OOF ile aynı çıktı; aşırı uyum işareti yok.
2. **En sağlam:** `s03alt_rankavg_8families.csv`, aynı 8 farklı model ailesinin (hedef kodlamalı / kodlamasız
   LightGBM, iki XGBoost, CatBoost, MLP, seyrek lojistik regresyon) **ağırlıksız** sıra ortalaması, CV 0.96153.
   Öğrenilmiş ağırlık yok; tek bir ailenin özel bir hatasına karşı en dayanıklı seçenek. 1. seçenekle sıra
   korelasyonu 0.9987.

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
