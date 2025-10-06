# Case Study: Evaluation of the nature of the optimal (new) candy (to sell)
## About
This project is a quick 1-2 evening exploration of a provided [dataset](https://github.com/fivethirtyeight/data/tree/master/candy-power-ranking), to perform a case study for an application process (and to test streamlit).
This repo should be taken as is. While it was fun to play around, I do not indeed on developing it further ;)

## Installation
Install everything needed from the `pyproject.toml` and run
```sh
streamlit run present.py
```

## Problems
##### Unknown tested distribution of candy
It is unknown how the tested candies where chosen. This leaves open:
- Are the chosen candies representative of their type/features
- Do the features represent the diversity and possibilities of candy well
While the "market research" probably cannot be performed on that many more candies (as people would not be reasonable be able to rank obscure/unknown products), it the limitation to existing sweets also limits the exploratory abilities of any conclusion. There is a survivors bias!

##### Small dataset size/ Lacking granularity
The dataset only provides a limited amount of features, that this evaluation also limits itself to.
Furthermore, the nature of the "winpercentage" favours uncontroversial popular sweets and does NOT allow the discovery of niche/ "controversial" markets.

##### Data not particularly useful for the task
This dataset heavily focuses on 