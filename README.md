# CAT
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
