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