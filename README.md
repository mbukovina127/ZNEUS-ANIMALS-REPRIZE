Image Classification [ANIMALS-10]
---

## 1. Prehľad datasetu

**Dataset:** https://www.kaggle.com/datasets/alessiocorrado99/animals10  
**Celkový počet vzoriek:** 26 179  
**Počet tried:** 10

Zastúpené triedy:  
`cane`, `cavallo`, `elefante`, `farfalla`, `gallina`, `gatto`, `mucca`, `pecora`, `ragno`, `scoiattolo`

 
Všetky súbory majú platné formáty (`jpeg`, `jpg`, `png`) a počas načítavania neboli zistené žiadne poškodené obrázky.

Triedy sú výrazne nerovnomerne distribuované:

- Najpočetnejšie triedy (**cane**, **ragno**) obsahujú približne 5 000 obrázkov.
- Najmenej zastúpená trieda (**elefante**) má iba 1 446 obrázkov.

Formát `.jpeg` tvorí 24 209 súborov, čo je najlepší vstupný formát pre spracovanie.

Obsahovo je dataset kvalitný, avšak obsahuje malé množstvo nesprávne zaradených obrázkov, čo je pri veľkých datasetoch ťažko identifikovateľné.

---

## 2. Rozmery obrázkov a pomer strán

- **Priemerný rozmer:** ~320 × 252 px  
- **Medián pomeru strán:** 1.31  
- **Q3:** ≤ 300 × 300 px  
- **Minimum:** 60 × 57 px  
- **Maximum:** 6720 × 6000 px  

Rozmery obrázkov nie sú normálne distribuované; množstvo vzoriek výrazne vybočuje z hlavného rozsahu.

### Outliery podľa rozmerov

| Typ outlierov | Počet | Popis |
|-------------------------------|-------|-------------------------------|
| Veľmi malé obrázky | 255 | Plocha < 38 850 px |
| Veľmi veľké obrázky | 1 970 | Plocha > 95 250 px |
| Extrémne úzky pomer strán | 261 | AR < 0.59 |
| Extrémne široký pomer strán | 186 | AR > 2.045 |
| **Spolu** | **2 672** | Vyžadujú predspracovanie |

---

## 3. Farebný model a distribúcia pixelov

Väčšina obrázkov je v modeli RGB s 8-bitovou hĺbkou, čo zodpovedá požiadavkám pre CNN architektúry.

Distribúcia pixelov je vo všeobecnosti vyvážená, avšak:

- **Modrý kanál vykazuje vyššiu smerodajnú odchýlku**.  
  Je to spôsobené obrázkami obsahujúcimi časti oblohy.

V extrémnych hodnotách sa nachádzajú takmer úplne biele alebo čierne obrázky (napr. transparentné PNG alebo obrázky s bielym/čiernym pozadím). Tieto vzorky nepredstavujú problém a môžu byť ponechané v datasete z dôvodu ľahkej extrakcii features.

---

## 4. Kľúčové zistenia z EDA

### Pozitíva
- Veľký počet vzoriek (26k) a vhodný počet tried (10)
- Žiadne chybné alebo nečitateľné obrázky
- Konzistentné farebné modely
- Správne formáty obrázkov
- Vysoká kvalita dát pre účely klasifikácie
- Svetlosť obrázkov nie je problém (ponechávame v datasete)

### Identifikované problémy
- Výrazná **nevyváženosť tried**
- 2 672 obrázkov s netypickými rozmermi (outliery)
- Mierna prítomnosť nesprávne zaradených vzoriek

---

## 5. Data split
- Kedže dataset obashoval veľa vzoriek, mohli sme si dovoliť rozdeliť dáta na:
  - Tréningová množina: 90% 
  - Testovacia množina: 10% (VAL/TEST)
  - BATCH SIZE: 64
- Model bude hodnotený iba na testovacej množine, ktorá nebude použitá počas tréningu ani validácie
- Hodnotenie je na základe troch kľúčových metrík: **Test Loss, Accuracy a F1-score** (štandardné metriky pre klasifikačné úlohy)

---

## 6. Data preprocessing
- Všetky obrázky boli zmenšené na **128x128px** pre konzistentný vstup do modelu (veľká majorita obrázkov mala rozmery okolo 300x250 px, takže zmenšenie nespôsobilo výraznú stratu features)
- Aplikovali sme **škálovanie pixelov** v rozsahu [0, 1] pre zlepšenie konvergencie modelu počas prevodu na tenzor
- Nomalizivali sme podla (mean,std) pre lepšiu komtaniblitu s ImageNet modelmi

---

## 7. Data augmentation
- Použili sme **data augmentation** techniky:
  - RandomResizedCrop(target_size) = náhodné vystrihnutie časti obrázka a jeho zmena na cieľovú veľkosť
  - RandomHorizontalFlip() = náhodné horizontálne prevrátenie obrázka
  - RandomRotation(10) = náhodné otočenie obrázka o ±10 stupňov
- Použili sme aj **minority-augmentation boost** pre vyváženie datasetu (viac augmentácie pre menej zastúpené triedy)
- Augemtovali sa iba tréningové dáta, validačné a testovacie dáta sme len zmenšovali