# AxiomaticDeepNets
paper Reference: https://arxiv.org/abs/1703.01365

I was able to interperate  images data classification (on pretrained googlenet )as well as IMDB data sentiment Analysis(on biderctional LSTM model) and visualize the results

Major modules implemented in the code

- Interpolation
- Gradient
- Integrated Gradient
- Visualization

## How to use code

### Process your Experiment(the image / text ) you want to explain as follows:

- upload your image in the /Images folder
- edit your config file with the path of the image ,result to be saved, pretrained model path and imagenet classes.txt in case of images
- run the main.py file 
- results are saved in the results folder representing visualizaion of : baseline , original image , gradient and Integrated Image, Overlay the image with IG

### Clone the repository

```git
git clone https://github.com/[username]/AxiomaticDeepNets.git
```

### Setup a new environment using `requirements.txt` in repo

```python
pip3 install -r requirements.txt 
```

### Setup configuration in `config.py` file

go to `src > config.py`

### Run `python main.py` with command-line arguments or with edited config file


```bash
python main.py 
```

### TODO
1. Improve documentation
