# HANDOFF: Kaggle Playground S6E10 (Predicting Airline Satisfaction) → NFL Big Data Bowl 2027

Kaynaklar: `README.md`, `experiments.md` (her koşu bir satır), `src/`, `kaggle/`, `logs/`, bu sohbet. `LESSONS.md` bir önceki
yarışmanın (S6E9) dersleri; bu yarışmaya girdi olarak kullanıldı. Kaydı olmayan sayı için "kayıt yok" yazıldı.

## 1. Problem, veri, metrik, son skorlar
- Sentetik tablo verisi: 22 özellik (4 kategorik, 13 hizmet puanı 0–5, Age, Flight Distance, 2 gecikme); train 699.635, test 299.844 satır,
  dış "orijinal" veri 129.880 satır (kurallar 2.6 izin veriyor). Hedef `satisfaction` (ikili), metrik ROC AUC, olasılık gönderilir.
- Son gönderim s15: **iç içe CV 0.962200**, **public LB 0.96180** (07.10.2026, 13/1087). Private LB: kayıt yok (yarışma 31.10.2026'da bitiyor).
- Kendi en iyi tek modeller (10 kat): RealMLP + te2 + orijinal satırlar 0.96148–0.96151, XGBoost FS3 0.96148. LightGBM varsayılan baseline 0.95780.

## 2. Pipeline
- **Temizlik:** tek eksik kolon Arrival Delay (train 292, test 130): GBDT'de NaN bırakıldı, sinir ağlarında 0 ile dolduruldu
  (`realmlp.py: twin_frame`). 4 kategorik → pandas category. Başka temizlik yapılmadı.
- **Özellik grupları** (`src/features.py: build`, kat içi kodlama anahtarları `te_keys`):
  `te1` her kolonun kat içi hedef kodlaması; `tefd` Flight Distance × diğer kolonlar; `te2` 20 düşük kardinaliteli kolonun bütün ikilileri;
  `fdprof` mesafe değeri başına etiketsiz profil (train+test); `cnt`/`cnt2` değer ve ikili sayımları; `inter` kategorik kesişimler;
  `opred` yalnız orijinal veride eğitilmiş LightGBM'in tahmini. FS1 = base+te1+tefd+fdprof+cnt+inter+orijinal satırlar; FS2 = FS1+te2; FS3 = FS2+cnt2+opred.
- **Modeller:** LightGBM / XGBoost (Optuna `t1`/`t2`), CatBoost, seyrek lojistik regresyon (`glm.py`), kolon-gömmeli MLP (`nn.py`),
  RealMLP ve TabM (pytabkit; görünümler pub/v4/v5/yk/ykv4, orijinal satırlarla), TabPFN-3.5 (5 kat, Kaggle 2×T4).
- **CV:** `StratifiedKFold(K, shuffle=True, random_state=42)`, K=5 (s01–s04), sonra 10; bütün modellerde aynı katlar, `seed=` yalnız model
  tohumu. Hedef kodlama: sklearn `TargetEncoder(cv=5)` her dış katın eğitim satırlarında (orijinal satırlar dahil). Erken durdurma doğrulama katında.
- **Ensemble:** s01–s07 kendi aileleri (`blend.py`: hill climbing / sıra ort. / olasılık ort. / logit LR, iç içe 5 kat). s08 ve sonrası
  `stack.py`: kendi + açık OOF üyelerinin logit'leri üstünde L2 lojistik regresyon (C=0.001), iç içe meta-CV (5 kat, tohum 7), kaynak ablasyonu.
- **Hesap:** yerel 4 çekirdek CPU; ağır sinir ağları/TabPFN Kaggle GPU kernel'larında (haftalık 30 saat kota); GPU'suz ek kapasite Kaggle CPU kernel'ları.

## 3. CV'yi en çok iyileştiren adımlar (büyükten küçüğe)
| Adım | Kazanç | Ölçüm bağlamı |
|---|---|---|
| `te1`: her kolonun kat içi hedef kodlaması (değer kimliği) | +0.00118 | LightGBM lr 0.1, 5 kat |
| MLP tohum ortalaması 1 → 7 | +0.0011 (0.95907 → 0.96017) | yalnız MLP; GBDT'de 2. tohum +0.00001–0.00003 |
| LightGBM varsayılan → 63 yaprak + erken durdurma | +0.00099 | 0.95780 → 0.95879 |
| Optuna ayarı + lr 0.1 → 0.03 | +0.00041 | LightGBM FS3 0.96089 → 0.96130 |
| Orijinal veriyi her eğitim katına ek satır (+`is_orig`) | +0.00031 | FS1 adımı; RealMLP pub'da +0.00016 (0.96110 → 0.96127) |
| `te2` (kolon ikilileri) | +0.00030 | LightGBM lr 0.1 |
| Farklı model ailelerini harmanlamak | +0.00030 | tek en iyi 0.96130 → istif 0.96160 (5 kat) |
| Kendi üyeler + açık OOF kütüphaneleri, L2 lojistik istif | +0.00029 | 0.96184 → 0.96213 (iç içe) |
| `tefd` (mesafe = rota kimliği × diğer kolonlar) | +0.00021 | LightGBM lr 0.1 |
| 5 → 10 kat | model başına +0.0001–0.0002 | harman 0.96165 → 0.96181 |
| Tablo temel modeli (TFM) OOF'ları (TabFM, LimiX2, TabICL2…) | +0.000030 | istifte; ablasyonda o kaynak çıkınca −0.000030 |
| TabM / `cnt2`+`opred` / hedef kodlamasız XGB / yekenot-görünümü RealMLP | +0.00003 / +0.00007 / +0.000008 / +0.000006 | istif veya FS2→FS3 |

**İşe yaramayanlar** (sayılar `experiments.md`/README'de): elle gecikme/puan özetleri ve grup ortalamaları (0.95843–0.95870, referans 0.95879);
0 bayrakları, puanların kategorik kopyası, `tecat`, `afill`; `omean` (te1 varken 0); orijinal satır ağırlığı 0.5/2 (0.96074 → 0.96063/0.96065);
`te3` üçlüler (gürültü); sayısal basamaklar 0.96041 / rota ortalamasından sapma 0.96015 / etiketsiz puan beklentisi 0.96035 (temel 0.96040);
TE girdili MLP harmanda ağırlık almadı (0.003); LightGBM rank_xendcg; bütün kolonlar kategorik (0.95995); segment başına istif ağırlığı;
kendi TabPFN (açık TabPFN ile korelasyon 0.998); hedef kodlamasız CatBoost / extra_trees (istifte 0); istifte probit(sıra) girdisi (0.961265),
negatif olmayan ağırlıklar (0.962006 / 0.962149), LightGBM meta-öğrenici (0.961997), logit kırpma ve sütun standardizasyonu (≤ 0.962151), C < 0.001.

## 4. Sızıntı kontrolleri ve CV–LB uyumu
- **Kat içi her şey:** hedeften türeyen her istatistik (TE) yalnız eğitim katında fit edildi; önbellek kat bazlı (`cache/<kolon>__o{orig}_k{K}_f{i}.npy`).
- **Düşmanca doğrulama** (`src/adversarial.py`): train/test AUC 0.500 (aynı dağılım); train/orijinal 0.834 → orijinal satırlara `is_orig` bayrağı.
- **İç içe ölçüm:** harman/istif ağırlıkları hep 4/5'te öğrenilip 1/5'te puanlandı. OOF içi ile iç içe skor aynı çıktı (s03 0.96159 / 0.96160).
- **Gürültü tabanı:** aynı özellik seti, 3 model tohumu → std ≈ 0.00005. 0.0001 altı farklar gürültü sayıldı.
- **Dış/açık OOF'lar** (`src/ext_pub.py`): id hizalama, şekil/sonluluk kontrolü, OOF AUC (0.90, 0.9625) dışında red. Her yeni kaynağın kodu
  okundu (fold içi TE, erken durdurmanın iç bölmede mi yapıldığı, etiketsiz özellik mi). Pickle dosyaları yüklenmedi; najiama 02/03
  (başka üyelerin harmanı) istif dışı. Kullanıcı kuralı: test satırlarını orijinal veriyle eşleştirip etiket kopyalamak yok.
- **Kendi bulduğumuz yanlılıklar:**
  - pytabkit en iyi epoch'u, ona verilen doğrulama katında (yani puanlanan satırlarda) seçiyor. Son epoch ile 0.96126, en iyi epoch ile
    0.96127/0.96128: yanlılık ~0.00001. `nn.py` de en iyi epoch'u doğrulama katında seçiyor; etkisi ölçülmedi (kayıt yok).
  - `opred` orijinal satırlarda örneklem içi tahmin: AUC 0.9983, çapraz tahmin 0.9952. Fark küçük olduğu için bırakıldı.
  - Test tahminleri kat ortalaması olduğu için OOF'tan yumuşak: istif skorunun "fark" bileşeni test'te OOF'un 0.967 katı. Etkisiz.
- **CV–LB:** 16 gönderimde iç içe CV ile public LB sıra korelasyonu 0.996. Public hep CV'nin altında:
  - yalnız kendi modellerimizle (s01–s07) aradaki fark 0.00054–0.00067;
  - açık OOF'lu istiflerde (s08–s15) fark 0.00040–0.00048;
  - örnekler: s01 → s02 CV +0.00033 / public +0.00028; s07 → s08 iç içe +0.00029 / public +0.00046.

  Public skorun std'si ≈ 0.001 (README tahmini), bu yüzden kararlar CV'ye göre verildi. Public sıralama bir günde 17 → 90 → 13 oynadı:
  46 takım aynı 0.96176'daydı, yani açık bir not defterinin kopyaları. Sıralama, skordan çok daha gürültülü bir sinyal.

## 5. Tekrar kullanılabilir kod
| Yol | Ne işe yarar |
|---|---|
| `src/common.py`: `load`, `folds`, `save` | veri yükleme; sabit tohumlu tabakalı katlar; OOF/test tahminini `oof/<etiket>.npy`, `preds/` altına yazma |
| `src/te_cache.py`: `te_block` | kat içi `TargetEncoder`; kolon × kat disk önbelleği, bellek için 40'lık parçalar, `TE_CACHE=0` ile bellek içi |
| `src/features.py`: `build`, `te_keys`, `aux_block`, `clean` | adlandırılmış özellik grupları (komut satırından `base,te1,...`); kodlama anahtarları; etiketsiz "beklenen değer" özellikleri |
| `src/train.py` | LightGBM/XGBoost/CatBoost K-kat CLI: hazır ayarlar, kat checkpoint'i, tamamlananı atlama, `experiments.md` satırı |
| `src/tune.py` | Optuna; sqlite'tan kaldığı yerden devam |
| `src/blend.py`: `fit_hill`, `fit_rank`, `fit_mean`, `fit_logit` | dört harman yöntemi, iç içe CV ile karşılaştırma |
| `src/stack.py`: `fit_lr`, `nested` | çok üyeli logit LR istifi, iç içe meta-CV, `ablate=1` ile kaynak çıkarma; `C`, `nonneg`, `clip`, `scale`, `tf` seçenekleri |
| `src/ext_pub.py`: `add`, `by_id` | dış OOF/test dosyalarını id ile hizalayıp doğrulayan yükleyici |
| `src/realmlp.py`: `twin_frame`, `orig_realmlp` | pytabkit RealMLP/TabM: her sayısal kolona kategorik ikiz, kat içi TE, dış satırlar, `best=0` (dürüst son epoch) |
| `src/nn.py`, `src/glm.py` | kolon-gömmeli PyTorch MLP; one-hot + ikililer üstünde seyrek lojistik regresyon |
| `src/adversarial.py`: `adv` | train/test ve train/dış veri ayrılabilirliği |
| `src/submit_check.py`: `write_submission`, `check` | gönderimi örnek dosya sırasıyla yazıp biçimini denetler |
| `kaggle/gpu_kernel.py`: `push`, `pushcpu`, `fetch` | `src/`'yi özel bir Kaggle not defterine paketleyip komutları GPU ya da CPU'da çalıştırır, `.npy` çıktıları geri indirir |
| `kaggle/tabpfn/build_kernel.py`, `worker.py`, `lean_patch.py` | TabPFN-3.5 tam kat bağlamı, iki GPU'lu zamanlayıcı, bellek-yalın KV önbelleği (hemingweb'in açık not defterinden uyarlama) |
| `scripts/meta_gbdt_probe.py`, `scripts/*.sh` | GBDT meta-öğrenici denemesi; yeniden başlatılabilir koşu kuyrukları |

## 6. NFL Big Data Bowl 2027'ye taşınabilirlik (~510 oyuncu, 5 pozisyon grubu, yorumlanabilirlik > skor, jüri + public notebook)
- [TAŞINIR] Sabit tohumlu katlar, her modelde aynı bölme: küçük n'de karşılaştırmanın adil olmasının tek yolu; pozisyon grubuna göre tabakala, oyuncu bazında böl.
- [TAŞINMAZ] 5 → 10 kat kazancı: +0.0001 büyük veride anlamlıydı; 510 oyuncuda tekrarlı K-kat ile skor varyansını göstermek daha önemli.
- [TAŞINIR] Tohum/tekrar gürültüsünü ölçüp eşiğin altını gürültü saymak: küçük n'de eşik çok daha büyük olacak; bootstrap güven aralığı yazıya girmeli.
- [TAŞINIR] İç içe CV ile model/ağırlık seçimini ölçmek: küçük örneklemde seçim yanlılığı daha da büyük.
- [TAŞINIR] Hedeften türeyen her istatistiği kat içinde hesaplamak (ör. pozisyon/takım ortalamaları): sızıntı hijyeni veri boyutundan bağımsız.
- [TAŞINMAZ] Değer kimliğini hedef kodlamak (`te1`/`tefd`, mesafe = rota): sentetik verinin üretim artefaktıydı.
- [TAŞINMAZ] `te2`/`te3` ikili-üçlü kodlamalar: 510 satırda hücreler çok seyrek kalır.
- [TAŞINIR] Etiketsiz grup profili özellikleri (`fdprof` mantığı): oyuncu metriğini kendi pozisyon grubunun profiline göre ifade etmek doğal ve okunur.
- [TAŞINMAZ] "Elle özet özellikler işe yaramadı" sonucu: burada ağaçlar ham kolonlardan öğrendi; 10 Hz izden özet çıkarmak (tepe hız, ivmelenme, yön değişimi) işin kendisi.
- [TAŞINIR] Grupları tek tek ekleyip gürültü eşiğine göre karar veren özellik ablasyonu: yazıda "hangi ölçüm neyi açıklıyor" tablosuna dönüşür.
- [TAŞINIR] Dış veriyi kaynak bayrağıyla ek satır olarak katmak ya da `opred` gibi tek bir skora indirmek: başka kohort/yıl verisi varsa; önce düşmanca doğrulama.
- [TAŞINIR] Düşmanca doğrulama: kombine yılları ya da gruplar arası kaymayı ucuza ölçer.
- [TAŞINMAZ] Optuna ile GBDT ayarı ve düşük lr: 510 satırda ayar CV'ye aşırı uyar; az parametreli, düzenlileştirilmiş modeller.
- [TAŞINMAZ] Doğrulama katında erken durdurma veya en iyi epoch seçimi: küçük katta OOF'u iyimser yapar; iç bölme ya da sabit tur sayısı kullan.
- [TAŞINMAZ] RealMLP / TabM / MLP: 510 örnek için fazla parametre, yorumlaması zor.
- [TAŞINIR] TabPFN'i karşılaştırma tavanı olarak kullanmak: küçük tablolar için tasarlandı, "yorumlanabilir model ne kadar geride" sorusunu cevaplar.
  Büyük bağlam altyapısı (`lean_patch`, iki GPU zamanlayıcısı) [TAŞINMAZ].
- [TAŞINMAZ] Yüzlerce üyeli lojistik istif ve açık OOF kütüphaneleri: liderlik tablosu ve OOF ekosistemi yok, yorumlanabilirlik önce.
- [TAŞINMAZ] Negatif ağırlıklı kontrast istifi ve AUC'ye özgü istif ayarları (probit, kırpma, C taraması): sadece LB skoru içindi.
- [TAŞINIR] "Ağırlığı en güçlü değil en farklı üye alır" dersi: farklı model ailelerinin hemfikir olduğu ilişkiler daha savunulabilir bulgudur.
- [TAŞINIR] Kaynak/grup ablasyonu (`stack.py ablate=1` mantığı): sensör/drill gruplarının katkısını bırak-bir-grup-dışarı ile raporlamak için doğrudan kullanılır.
- [TAŞINIR] Dış kod ve veriyi kullanmadan önce okumak, pickle yüklememek: public notebook teslim edileceği için kaynak gösterme de gerekecek.
- [TAŞINIR] `experiments.md`'ye her koşu için tek satır, OOF dosyaları, checkpoint'ler: yazıdaki her sayının izlenebilir olması jüri için önemli.
- [TAŞINIR] `kaggle/gpu_kernel.py` ile yerel kodu Kaggle not defterinde çalıştırmak: public notebook'un uçtan uca orada çalıştığını doğrular; GPU gerekmez (`pushcpu`).
- [TAŞINMAZ] İki final adayı seçme stratejisi: LB yok; tek, savunulabilir model ve duyarlılık analizi yeterli.
