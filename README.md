# Class conditioned text generation with style attention mechanism for embracing diversity
### 🔄️ To install python dependencies before train:
```
sh setup.sh
```

### 1️⃣ To train and inference model:
#### 1. Run classifier
```
python classifier.py
```

#### 2. Run preliminary module  
```
python recon_main.py --mode train
```

#### 3. Run adversarial module
```
python adv_main.py --mode train
```

#### 4. Run style attention module
```
python style_attn_main.py --mode train
```

#### 5. Run test version of the style attention module
```
python style_attn_main.py --mode test
```

***

### 🔄 To get file for inference:
You can run inference just by downloading this file
```
sh download.sh
```

### 2️⃣ To inference model using the file downloaded above:
```
python style_attn_main.py --mode test --d [Amazon|YELP]
```

***

### ☑ To get target style sentence:
```
1. Type sentence to "Input Sentence: "
2. Input style range from 1 to 5 to "Target Style: "
```

### 📃 To cite the paper:
```
@article{KWON2024111893,
title = {Class conditioned text generation with style attention mechanism for embracing diversity},
journal = {Applied Soft Computing},
volume = {163},
pages = {111893},
year = {2024},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2024.111893},
url = {https://www.sciencedirect.com/science/article/pii/S1568494624006677},
author = {Naae Kwon and Yuenkyung Yoo and Byunghan Lee},
keywords = {Natural language generation, Text style, Multi-class, Style attention, Non-parallel},
abstract = {In the field of artificial intelligence and natural language processing (NLP), natural language generation (NLG) has significantly advanced. Its primary aim is to automatically generate text in a manner resembling human language. Traditional text generation has mainly focused on binary style transfers, limiting the scope to simple transformations between positive and negative tones or between modern and ancient styles. However, accommodating style diversity in real scenarios presents greater complexity and demand. Existing methods usually fail to capture the richness of diverse styles, hindering their utility in practical applications. To address these limitations, we propose a multi-class conditioned text generation model. We overcome previous constraints by utilizing a transformer-based decoder equipped with adversarial networks and style-attention mechanisms to model various styles in multi-class text. According to our experimental results, the proposed model achieved better performance compared to the alternatives on multi-class text generation tasks in terms of diversity while it preserves fluency. We expect that our study will help researchers not only train their models but also build simulated multi-class text datasets for further research.}
}
```

***
