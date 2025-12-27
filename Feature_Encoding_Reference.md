# Demografische Features - Encoding Referenz

## Übersicht der 9 Features

Diese Features werden für das Clustering verwendet (in `feats` Liste).

---

## 1. Province_Encoded

**Original:** Anzahl Kunden in der Provinz (Frequency Encoding)

| Z-Score  | Bedeutung                                 |
| -------- | ----------------------------------------- |
| Z < -1.5 | Sehr kleine Provinz (sehr wenige Kunden)  |
| Z ≈ -1.0 | Kleine Provinz (unterdurchschnittlich)    |
| Z ≈ 0.0  | Durchschnittliche Provinzgröße            |
| Z ≈ +1.0 | Große Provinz (überdurchschnittlich)      |
| Z > +1.5 | Sehr große Provinz (z.B. Ontario, Quebec) |

---

## 2. City_Encoded

**Original:** Anzahl Kunden in der Stadt (Frequency Encoding)

| Z-Score  | Bedeutung                                 |
| -------- | ----------------------------------------- |
| Z < -1.5 | Sehr kleine Stadt (sehr wenige Kunden)    |
| Z ≈ -1.0 | Kleine Stadt (unterdurchschnittlich)      |
| Z ≈ 0.0  | Durchschnittliche Stadtgröße              |
| Z ≈ +1.0 | Große Stadt (überdurchschnittlich)        |
| Z > +1.5 | Sehr große Stadt (z.B. Toronto, Montreal) |

---

## 3. FSA_Encoded

**Original:** Anzahl Kunden im Postleitzahl-Gebiet (Frequency Encoding)

- FSA = Forward Sortation Area (erste 3 Zeichen der Postleitzahl)

| Z-Score  | Bedeutung                       |
| -------- | ------------------------------- |
| Z < -1.5 | Sehr dünn besiedeltes Gebiet    |
| Z ≈ -1.0 | Unterdurchschnittlich besiedelt |
| Z ≈ 0.0  | Durchschnittliches Gebiet       |
| Z ≈ +1.0 | Überdurchschnittlich besiedelt  |
| Z > +1.5 | Sehr dicht besiedeltes Gebiet   |

---

## 4. Gender_Encoded

**Original:** 0 = Female, 1 = Male

| Z-Score  | Bedeutung                                             |
| -------- | ----------------------------------------------------- |
| Z < -1.5 | Fast ausschließlich Frauen (sehr hoher Female-Anteil) |
| Z ≈ -1.0 | Überwiegend Frauen (~65% Female)                      |
| Z ≈ 0.0  | Ausgewogenes Geschlechterverhältnis (50/50)           |
| Z ≈ +1.0 | Überwiegend Männer (~65% Male)                        |
| Z > +1.5 | Fast ausschließlich Männer (sehr hoher Male-Anteil)   |

---

## 5. Education_Level_Num

**Original:** 0 = Low, 1 = Mid, 2 = High

**Mapping:**

- **0 = Low:** High School or Below
- **1 = Mid:** College, Bachelor
- **2 = High:** Master, Doctor

| Z-Score  | Bedeutung                                        |
| -------- | ------------------------------------------------ |
| Z < -1.5 | Ausschließlich Low Education (High School)       |
| Z ≈ -1.0 | Überwiegend Low Education                        |
| Z ≈ 0.0  | Durchschnittliches Bildungsniveau (Mix)          |
| Z ≈ +1.0 | Überwiegend Mid-High Education (Bachelor/Master) |
| Z > +1.5 | Ausschließlich High Education (Master/Doctor)    |

---

## 6. Location_Code_Num

**Original:** 0 = Rural, 1 = Suburban, 2 = Urban

| Z-Score  | Bedeutung                        |
| -------- | -------------------------------- |
| Z < -1.5 | Ausschließlich Rural (ländlich)  |
| Z ≈ -1.0 | Überwiegend Rural/Suburban       |
| Z ≈ 0.0  | Mix aus Rural/Suburban/Urban     |
| Z ≈ +1.0 | Überwiegend Suburban/Urban       |
| Z > +1.5 | Ausschließlich Urban (städtisch) |

---

## 7. Income_Bin_Num

**Original:** 0 = Low, 1 = Medium, 2 = High, 3 = Very High, 4 = Ultra High

**Bins:** 5 equal-width Einkommensbereiche

| Z-Score  | Bedeutung                             |
| -------- | ------------------------------------- |
| Z < -1.5 | Ausschließlich Low Income             |
| Z ≈ -1.0 | Überwiegend Low-Medium Income         |
| Z ≈ 0.0  | Durchschnittliches Einkommen (Medium) |
| Z ≈ +1.0 | Überwiegend High-Very High Income     |
| Z > +1.5 | Ausschließlich Ultra High Income      |

---

## 8. Marital_Divorced

**Original:** 1 = Divorced, 0 = Not Divorced (One-Hot Encoded)

| Z-Score  | Bedeutung                                            |
| -------- | ---------------------------------------------------- |
| Z < -1.0 | Niemand oder fast niemand geschieden (0-5% Divorced) |
| Z ≈ -0.5 | Wenige Geschiedene (unterdurchschnittlich)           |
| Z ≈ 0.0  | Durchschnittlicher Anteil Geschiedener               |
| Z ≈ +0.5 | Überdurchschnittlich viele Geschiedene               |
| Z > +1.0 | Sehr viele Geschiedene (>80% Divorced)               |

---

## 9. Marital_Married

**Original:** 1 = Married, 0 = Not Married (One-Hot Encoded)

| Z-Score  | Bedeutung                                            |
| -------- | ---------------------------------------------------- |
| Z < -1.0 | Niemand oder fast niemand verheiratet (0-5% Married) |
| Z ≈ -0.5 | Wenige Verheiratete (unterdurchschnittlich)          |
| Z ≈ 0.0  | Durchschnittlicher Anteil Verheirateter              |
| Z ≈ +0.5 | Überdurchschnittlich viele Verheiratete              |
| Z > +1.0 | Sehr viele Verheiratete (>80% Married)               |

---

## Wichtige Hinweise zur Interpretation

### Z-Score Faustregel

- **|Z| < 0.5:** Nahe am Durchschnitt (unauffällig)
- **0.5 < |Z| < 1.0:** Leicht über/unter Durchschnitt
- **1.0 < |Z| < 2.0:** Deutlich über/unter Durchschnitt (wichtiges Merkmal!)
- **|Z| > 2.0:** Extrem über/unter Durchschnitt (sehr charakteristisch!)

### Skalierung

Alle Features wurden mit **StandardScaler** (Z-Score Normalization) skaliert:

- **Mean (μ) = 0**
- **Std (σ) = 1**

### Marital Status

⚠️ **Marital_Single** wurde entfernt (redundant), da es durch die anderen beiden Variablen bestimmt ist:

- Wenn `Marital_Married = 0` und `Marital_Divorced = 0` → dann ist Person Single

### Frequency Encoding

Features **Province_Encoded**, **City_Encoded** und **FSA_Encoded** verwenden Frequency Encoding:

- Höhere Werte = mehr Kunden aus dieser Region
- Niedrige Werte = wenige Kunden aus dieser Region
- Diese Features erfassen geografische Konzentration

---

## Anwendung in Cluster-Profilen

### Beispiel-Interpretation einer Heatmap-Zeile:

| Feature             | Z-Score | Interpretation                                  |
| ------------------- | ------- | ----------------------------------------------- |
| Province_Encoded    | +1.8    | Cluster aus großer Provinz (z.B. Ontario)       |
| City_Encoded        | +2.1    | Cluster aus Großstadt (z.B. Toronto)            |
| FSA_Encoded         | +1.5    | Dicht besiedeltes Gebiet                        |
| Gender_Encoded      | +0.8    | Etwas mehr Männer (~60%)                        |
| Education_Level_Num | +1.2    | Überdurchschnittlich gebildet (Bachelor/Master) |
| Location_Code_Num   | +1.6    | Überwiegend urban                               |
| Income_Bin_Num      | +1.4    | Hohes Einkommen (High-Very High)                |
| Marital_Divorced    | -0.3    | Wenige Geschiedene                              |
| Marital_Married     | +0.9    | Viele Verheiratete                              |

**→ Typisches Profil:** Gebildete, gut verdienende, verheiratete Männer aus urbanen Großstädten in großen Provinzen.

---

## Weitere Informationen

- **Notebook:** `Group80_Clustering_Code.ipynb`
- **Encoding:** Zelle 48
- **Skalierung:** Zelle 73
- **Feature Selection:** Zelle 62, 66
