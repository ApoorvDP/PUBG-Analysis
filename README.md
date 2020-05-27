# PUBG-Analysis
The purpose of this analysis is to predict a player's survival time based on certain input parameters. The longer the survival time, the better the outcome for the player.

Players can use this code to evaluate their game strategy; they can input appropriate parameters and then see how long they are likely to survive.

The guide to the repository code is in `readme.txt` within the repository.

## Dataset
The dataset used for training the model for this project was obtained from [Kaggle](https://www.kaggle.com/skihikingkevin/pubg-match-deaths). It contained aggregated data of about 720,000 matches in PUBG and records of over 65 million deaths in said matches. The dataset consists of two parts, data from the matches and details of various deaths split into multiple CSV files of two types, aggregate and deaths. The total uncompressed size of the data is approximately 19 GB. In the “deaths” subset of the data, the files recorded every player death that occurred in the 720,000 matches. Each row captured information about an event where a player died within the match. In the “aggregate” files, each match's meta information and player statistics were summarized (as provided by PUBG).

While working with the data, we saw that there were a few outliers in the dataset, wherein some players have survival times in the range of millions when it should ideally be below 2000. The results for omitting records with survival time below 2000 and 2500 were the same so the larger threshold of 2500 was selected to be on the safe side.

## Methodology
From a machine learning perspective, an enormous dataset would take forever to be analyzed on a single commodity machine. Hence, performing machine learning on large datasets necessitates a distributed set-up to achieve manageable load balancing across machines and reasonable model training times.

The machine learning was implemented using distributed PyTorch. The dataset was stored in an HDFS cluster with over 30 worker nodes and a min replication of 3 for the data files.

Taking a closer look at the data set schema, only the “aggregate” subset has data relevant to machine learning. The fields relevant to player survival are: _game_size_, _match_mode_, _party_size_ and _player_dmg_. The aforementioned four fields are considered inputs for the neural network, and thereby extracted from the “aggregate” subset along with _player_survive_time_, which serves as the target field for the neural network.

The data was split 80:20 into training and testing subsets. The neural network code was written using PyTorch. The _torch.dist_ library was used for distributing the execution of the neural network across multiple machines. _InsecureClient_ within the _hdfs_ library was used to connect with HDFS. The network was trained on 12 CUDA-enabled machines.

## Result
The network achieved a test RMSE of only 2.796e-09, which can be construed as accurate given the typical order of survival durations.
