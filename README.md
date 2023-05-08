# ReLU närvivõrgu ennustusmääramatuse hindamine regressioonüles

## Struktuur
Töös kasutatav Pythoni versioon oli 3.9.1. Loodava environementi jooksutamiseks tuleb installida vajalikud Pythoni teegid, käsiuga ``pip install -r requirements.txt``

## Koodirepositooriumi ülesehitus
Kaustas ``notebooks`` on Jupyteri notebookid, mida kasutati töös andmete analüüsimiseks. Kaustas ``katsetused`` on Jupyteri notebookid, millest on näha töö protsessi, kuid nad vältimatult ei käivitu.
Kaustas ``scripts`` on skriptid, mida läheb vaja teiste asjade jooksutamiseks. Kodukaustas asuvad ``.sh`` failid jooksutavad kõiki ``.job`` SLURM töösid, mis tõmbavad vajalikud treenimisskriptid tööle.

Kaust ``andmed`` on see kuhu kõik skriptid toodavad oma andmeid ning ``plots`` see kuhu pildid salvestatakse.

HPC-s jooksutamiseks on loodud SLURM jobid, nende mugavamaks jooksutamiseks on bash skriptid, mis panevad kõik vajalikd SLURMi tööd tööle.