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

Rozmery obrázkov nie sú normálne distribuované, množstvo vzoriek výrazne vybočuje z hlavného rozsahu.

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

Väčšina obrázkov je v modeli RGB s 8-bitovou hĺbkou, čo zodpovedá požiadavkám pre CNN.

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

--- 
## 8. Configuration
Set-up
```python
EPOCHS = 100
LEARNING_RATE = 0.1

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(img_class_model.parameters(), lr=0.2, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=4)
e_stop = EarlyStopping(patience=5, diff=0.01)
```   

Toto je podoba finalneho modelu
```python 
class ImageClassifier(nn.Module):
    def __init__(self, classes: int, dropout: nn.Dropout = nn.Dropout(0.3)):
        super(ImageClassifier, self).__init__()
        self.numberOfClasses = classes
        self.dropout = dropout

        # Block 1: 256 -> 128
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Block 2: 128 -> 64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Block 3: 64 -> 32
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Block 4: 32 -> 16 (Grad-CAM)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Global average pooling reduces (256×16×16) to (256)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, self.numberOfClasses)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = fun.leaky_relu(self.fc1(x))
        return self.fc2(x)
```

---

## 9. Experimentácia
>Prvé verzii modelu mali iba 3 až 4 konvolučné vrstvy ktoré dosahovali výsledky 60% presnosti.
>
>![img.png](img/first_training.png)

>Naším prvotným riešením bolo použiť dropout na zmenšenie závislosti modelu na jednotlivé neuróny.
>Toto riešenie mierne zvýšilo presnosť modelu no po overení s heatmap-ou vygenerovanou pomocou grad-camu sme zistili že model sa priveľmi spolieha na charakteristicke črty zvierat ako srsť, textúra, srste na rôzne sfarbenia srste.
>
>![img.png](img/droup-out_training.png)

>Vďaka tomuto sme zistili že najlepším riešením je pridať ďalšie konvolučné vrstvy.
>To nám opäť zvýšili presnosť modelu na hodnoty okolo 75% ale čo bolo dôležitejšie je že model už nenarazil na horný limit tak ako pri predošlých verziách.
>
>![img.png](img/more_conv_training.png)

>Po overením s heatmap-ou sme zistili že sa model už nerozhoduje podľa hyperšpecifických charakteristík no stále sa nezameriava na celé zviera.
>Naším finálnym riešením bolo pridanie viacero konovlučnych sieti do jedneho bloku aby sme nestrácali detaily priskoro a taktiež použitie viacero rôznych kerneloch na dosiahnutie toho aby sa model dokázal naučiť rozsiahleši kontext z obrázkov. 
>Po dlhšom trenovaní sme dosiahli vysledky v blízkosti 90%(91% trenovacia presnosť, 89% validačna presnosť a 87% testovacia) 
> 
>![img.png](img/best_training.png)
>![img.png](img/best_val.png)

### 9.1 Zistenia z GRAD-CAM
![](findings/squirel%20based%20on%20tree.png)
![](findings/100e_badsquirel.png)
![](findings/squirel_is_not_an_elefant.png)
>Veľký dôraz na textúru alebo na prvky ktore nemajú nič spoločne so zvieraťom

### 9.2 Porovnávanie GRAD-CAM
Nasledne zobrazíme ako ovplivnňuje zvolený block GRAD-CAMu výstup heatmapy

![block1.png](img/block1.png)
> Blok 1 zachytáva hrany a prechody ako vidime nemajú veľký vplyv na výsledok


![block2.png](img/block2.png)
> Blok 2 sa naučil zachytávať farby a menšie tvary


![block3.png](img/block3.png)
> Blok 3 už má dokáže pochopyť súvis medzi časťami v kratšej vzdialenosti
 
![block4.png](img/block4.png)
> Block 4 už berie do kontextu cele objekty ako je hlava mačky.

---

## 10. Results and evaluation metrics 
![results.png](img/results.png)
>Finálne skóre testovacieho behu bolo: \
Test_loss: 0.4214895398630572 \
Test_Acc: 0.8761354252683733 \
Test_F1: 0.8633964159391541

### 10.1 Confusion matrix
![confusion matrix](img/conf_mat.png)
> Pomocou confusion matrix sme si overili naše tušenie, že klasifikacia podobne vyzerajúcich druhov je najmenej presná.  


---

## 11. Summary 
Model sa naučil klasifikovať obrázky zvierat s vysokou presnosťou (**87.6% na testovacej množine**).

### Kľúčové zistenia:
#### 1. Bias Modelu
   - Všetky triedy boli klasifikované s úspešnosťou nad 80 %, takže model nevykazuje výrazný bias voči žiadnej kategórii, čo bola pôvodne jedna z obáv.
   - Data augmentation a minority-boost bolo správne
#### 2. Features Extraction
  - Počas analýzy pomocou **Grad-CAM** sa však ukázalo, že model sa v niektorých prípadoch sa vytvorili neuróny, ktoré sa zameriavajú skôr na **textúry a pozadie** namiesto dominantného objektu. 
  - Tento problém by bolo možné riešiť odstránením alebo neutralizovaním pozadia, čo je však náročnejší krok.
  - Pri obrázkoch s čiernym alebo bielym pozadím model fungoval najlepšie
#### 3. Misclassifications
-   Zaujímave zistenie je že chyby nie sú komutatívne, teda šanca že model chybne označí mačku ako psa je 5-krát väčšia ako v opačnom prípade. 
-   Časté zámeny psa za koňa, kravu alebo mačku (nie naopak)
  - Podobne si model mýlil kopytníky (kôň, krava, ovca) medzi sebou, čo sú triedy s prirodzene blízkymi vizuálnymi znakmi.