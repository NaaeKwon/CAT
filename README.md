# CAT
### 🔄️ To install setup:
```
conda install (your_env_name) env32bit --file set-file.txt
```

### 🔄 To download .pt .binary file:
Due to the size limit, download files at the url of the following file
```
pt_file_url.txt
amazon_bin_file_url.txt
yelp_bin_file_url.txt
```


### 1️⃣ Preliminary module  
### To see usage:
```
python recon_main.py --help
```


### 2️⃣ Adversarial module
Initialize parameter with reconstruction model (Recon.pt)
### To see usage:
```
python adv_main.py --help
```


### 3️⃣ Style attention module
Initialize parameter with adversarial module (Adv.pt)
### To see usage:
```
python style_attn_main.py --help
```
### To get target style sentence:
```
1. Run the code in test option 
2. Type sentence to "Input Sentence: "
3. Input Style range from 1 to 5 to "Target Style: "
```
