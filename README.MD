# Recommender Systems challenge - Politecnico di Milano, 2022
In this repository there is the code I used for the challenge of the Recommender Systems course at Politecnico di Milano.

The goal of the competition was to create the recommender system TV programs by providing 10 recommended products to each target user.
[Link to the official website of the challenge](https://www.kaggle.com/competitions/recommender-system-2022-challenge-polimi)

## Evaluation
... score and position

## Data
description of the data

## Reccomender
description of reccomender

## Setup
- Download the data from [Kaggle](https://www.kaggle.com/competitions/recommender-system-2022-challenge-polimi/data)
- Install packages with [Poetry](https://python-poetry.org/docs/basic-usage/#installing-dependencies)
- Download the data in the `Data` folder from [the challange webpage](https://www.kaggle.com/competitions/recommender-system-2022-challenge-polimi/data)

## Credits
The code in this repository is inspired by [MaurizioFD/RecSys_Course_AT_PoliMi](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi), a repository used during the Recommender Systems course at Politecnico di Milano.


RP3 Best MAP so far: 0.023999901625354463
base=12.62329790627502, views_weight=82.89881321660107, details_weight=27.811514733648256

KNN: Best MAP so far: 0.02230552800684151
base=46.69485845326952, views_weight=78.51759613930136, details_weight=19.967378215835975

EASE_R: Best MAP so far: 0.02633010813203595
base=81.12913520607813, views_weight=89.60912999234932, details_weight=31.800347497186387


NMF -0.012
PureSVD -0.012
AsySVD too slow (10hr+) and FAIL
SLIM BPR 0.020
MF BPR very low score
SLIM ElasticNet -0.0244 {'topK': 979, 'l1_ratio': 0.0018154743396116312, 'alpha': 0.0024650504717633574} (untuned URM)
SLIM ElasticNet tuned URM -0.027 base=83.7264755594732, views_weight=93.47717420378852, details_weight=71.2945046148223 
SLIM ElasticNet tuned URM -> retuning -0.271
hybrid IALS EASEr KNN RP3 -0.0272
hybrid IALS EASEr KNN RP3 SLIM ?
FunkSVD fail
MultVAE too long (10hr+)
MF_SDVpp very low score