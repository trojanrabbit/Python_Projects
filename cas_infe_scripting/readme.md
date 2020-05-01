Readme für ```solution_project_2.py```

## Autor
Simon Würsten, November 2019

## Datenquellen
1. https://data.sbb.ch/explore/dataset/passagierfrequenz/
1. https://data.sbb.ch/explore/dataset/haltestelle-offnungszeiten

## Packages

```<module>.__version__``` ergibt folgende Versionsnummer:
- urllib (Meldung *module 'urllib' has no attribute '\_\_version\_\_'*)
- json 2.0.9
- pandas 0.24.2
- mathplotlib.pyplot (Meldung *module 'mathplotlib.pylot' has no attribute '\_\_version\_\_'*)
- numpy 1.16.4

## Erläuterungen zur nachgebesserten Version 2_2
Sobald ich die Daten mit einer Funktion hole und mit ```return``` zurückgebe, kann ich die JSON-Daten im nachfolgenden Codeabschnitt nicht mehr parsen mit ```json.loads()``` und zu einem Dictionary umwandeln. Dies ist im Code so vermerkt. Ich habe keine Ahnung wieso das so ist.
Die Fehlermeldung lautet: *TypeError: the JSON object must be str, bytes or bytearray, not list*
